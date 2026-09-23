import * as fs from 'fs/promises';
import * as os from 'os';
import * as path from 'path';
import * as crypto from 'crypto';
import { Context } from './context';
import { Embedding, EmbeddingVector } from './embedding';
import { Splitter, CodeChunk } from './splitter';
import { VectorDatabase } from './vectordb';

// Mirrors Context's private generateId() so tests can construct fixture rows
// that line up with the IDs the code under test will compute.
function generateId(relativePath: string, startLine: number, endLine: number, content: string): string {
    const combinedString = `${relativePath}:${startLine}:${endLine}:${content}`;
    const hash = crypto.createHash('sha256').update(combinedString, 'utf-8').digest('hex');
    return `chunk_${hash.substring(0, 16)}`;
}

class TestEmbedding extends Embedding {
    protected maxTokens = 8192;

    async detectDimension(): Promise<number> {
        return 3;
    }

    async embed(_text: string): Promise<EmbeddingVector> {
        return { vector: [1, 0, 0], dimension: 3 };
    }

    async embedBatch(texts: string[]): Promise<EmbeddingVector[]> {
        return texts.map(() => ({ vector: [1, 0, 0], dimension: 3 }));
    }

    getDimension(): number {
        return 3;
    }

    getProvider(): string {
        return 'test';
    }
}

// Always splits a file into exactly two deterministic chunks, so tests can
// exercise the "existing IDs are a strict subset" and "existing IDs differ"
// resume scenarios without needing a real AST/LangChain splitter.
class TwoChunkSplitter implements Splitter {
    async split(code: string, language: string, filePath?: string): Promise<CodeChunk[]> {
        return [
            { content: `${code}::chunk1`, metadata: { startLine: 1, endLine: 1, language, filePath } },
            { content: `${code}::chunk2`, metadata: { startLine: 2, endLine: 2, language, filePath } },
        ];
    }

    setChunkSize(): void { }
    setChunkOverlap(): void { }
}

type ExistingChunksByPath = Record<string, string[]>;

const createVectorDatabase = (existingChunksByPath: ExistingChunksByPath = {}): jest.Mocked<VectorDatabase> => ({
    createCollection: jest.fn().mockResolvedValue(undefined),
    createHybridCollection: jest.fn().mockResolvedValue(undefined),
    dropCollection: jest.fn().mockResolvedValue(undefined),
    hasCollection: jest.fn().mockResolvedValue(false),
    listCollections: jest.fn().mockResolvedValue([]),
    insert: jest.fn().mockResolvedValue(undefined),
    insertHybrid: jest.fn().mockResolvedValue(undefined),
    search: jest.fn().mockResolvedValue([]),
    hybridSearch: jest.fn().mockResolvedValue([]),
    delete: jest.fn().mockResolvedValue(undefined),
    query: jest.fn(async (_collectionName: string, filter: string) => {
        for (const [relativePath, ids] of Object.entries(existingChunksByPath)) {
            if (filter.includes(`relativePath == "${relativePath}"`)) {
                return ids.map(id => ({ id }));
            }
        }
        return [];
    }) as any,
    getCollectionDescription: jest.fn().mockResolvedValue(''),
    checkCollectionLimit: jest.fn().mockResolvedValue(true),
    getCollectionRowCount: jest.fn().mockResolvedValue(0),
    getCollectionDimension: jest.fn().mockResolvedValue(3),
});

describe('Context indexCodebase resume support', () => {
    let tempRoot: string;
    let originalHome: string | undefined;
    let originalHybridMode: string | undefined;

    beforeEach(async () => {
        tempRoot = await fs.mkdtemp(path.join(os.tmpdir(), 'claude-context-resume-'));
        const homeDir = path.join(tempRoot, 'home');
        await fs.mkdir(homeDir, { recursive: true });
        originalHome = process.env.HOME;
        originalHybridMode = process.env.HYBRID_MODE;
        process.env.HOME = homeDir;
        process.env.HYBRID_MODE = 'false';
    });

    afterEach(async () => {
        if (originalHome === undefined) {
            delete process.env.HOME;
        } else {
            process.env.HOME = originalHome;
        }
        if (originalHybridMode === undefined) {
            delete process.env.HYBRID_MODE;
        } else {
            process.env.HYBRID_MODE = originalHybridMode;
        }
        await fs.rm(tempRoot, { recursive: true, force: true });
    });

    async function createProject(files: Record<string, string>): Promise<string> {
        const project = path.join(tempRoot, 'project');
        await fs.mkdir(project, { recursive: true });
        for (const [name, content] of Object.entries(files)) {
            await fs.writeFile(path.join(project, name), content);
        }
        return project;
    }

    it('skips a fully indexed, unchanged file: no embedding, no insert', async () => {
        const content = 'const unchanged = 1;';
        const project = await createProject({ 'unchanged.ts': content });

        const id1 = generateId('unchanged.ts', 1, 1, `${content}::chunk1`);
        const id2 = generateId('unchanged.ts', 2, 2, `${content}::chunk2`);

        const vectorDatabase = createVectorDatabase({ 'unchanged.ts': [id1, id2] });
        vectorDatabase.hasCollection.mockResolvedValue(true);

        const embedding = new TestEmbedding();
        const embedBatchSpy = jest.spyOn(embedding, 'embedBatch');
        const context = new Context({
            embedding,
            vectorDatabase,
            codeSplitter: new TwoChunkSplitter(),
        });

        const collectionName = context.getCollectionName(project);
        const stats = await context.indexCodebase(project);

        expect(stats.indexedFiles).toBe(1);
        expect(stats.status).toBe('completed');
        expect(vectorDatabase.query).toHaveBeenCalled();
        // Milvus defaults a filtered query with no explicit limit to 100 rows,
        // which truncated the chunk-ID lookup for files with more than 100
        // chunks. The lookup must always pass Milvus's own per-query maximum.
        expect(vectorDatabase.query).toHaveBeenCalledWith(
            collectionName,
            'relativePath == "unchanged.ts"',
            ['id'],
            16384
        );
        expect(embedBatchSpy).not.toHaveBeenCalled();
        expect(vectorDatabase.insert).not.toHaveBeenCalled();
        expect(vectorDatabase.delete).not.toHaveBeenCalled();
    });

    it('re-indexes a partially indexed file: deletes stale chunks, then embeds and inserts all chunks', async () => {
        const content = 'const partial = 1;';
        const project = await createProject({ 'partial.ts': content });

        const id1 = generateId('partial.ts', 1, 1, `${content}::chunk1`);
        const id2 = generateId('partial.ts', 2, 2, `${content}::chunk2`);

        // Only one of the two chunks made it into the collection before the
        // previous run was interrupted: existing IDs are a strict subset.
        const vectorDatabase = createVectorDatabase({ 'partial.ts': [id1] });
        vectorDatabase.hasCollection.mockResolvedValue(true);

        const embedding = new TestEmbedding();
        const embedBatchSpy = jest.spyOn(embedding, 'embedBatch');
        const context = new Context({
            embedding,
            vectorDatabase,
            codeSplitter: new TwoChunkSplitter(),
        });

        const collectionName = context.getCollectionName(project);
        const stats = await context.indexCodebase(project);

        expect(stats.indexedFiles).toBe(1);
        expect(vectorDatabase.delete).toHaveBeenCalledWith(collectionName, [id1]);
        expect(embedBatchSpy).toHaveBeenCalledWith([`${content}::chunk1`, `${content}::chunk2`]);

        const insertedIds = vectorDatabase.insert.mock.calls
            .flatMap(([, documents]) => documents)
            .map((doc: any) => doc.id)
            .sort();
        expect(insertedIds).toEqual([id1, id2].sort());
    });

    it('re-indexes a changed file: deletes the non-matching chunk, then embeds and inserts the new chunks', async () => {
        const content = 'const changed = 1;';
        const project = await createProject({ 'changed.ts': content });

        // Simulates content/splitter settings having changed since the last
        // run: the previously stored chunk ID does not appear in the new set.
        const staleId = 'chunk_deadbeefcafebabe';
        const id1 = generateId('changed.ts', 1, 1, `${content}::chunk1`);
        const id2 = generateId('changed.ts', 2, 2, `${content}::chunk2`);

        const vectorDatabase = createVectorDatabase({ 'changed.ts': [staleId] });
        vectorDatabase.hasCollection.mockResolvedValue(true);

        const embedding = new TestEmbedding();
        const embedBatchSpy = jest.spyOn(embedding, 'embedBatch');
        const context = new Context({
            embedding,
            vectorDatabase,
            codeSplitter: new TwoChunkSplitter(),
        });

        const collectionName = context.getCollectionName(project);
        const stats = await context.indexCodebase(project);

        expect(stats.indexedFiles).toBe(1);
        expect(vectorDatabase.delete).toHaveBeenCalledWith(collectionName, [staleId]);
        expect(embedBatchSpy).toHaveBeenCalledWith([`${content}::chunk1`, `${content}::chunk2`]);

        const insertedIds = vectorDatabase.insert.mock.calls
            .flatMap(([, documents]) => documents)
            .map((doc: any) => doc.id)
            .sort();
        expect(insertedIds).toEqual([id1, id2].sort());
    });

    it('throws a clear dimension-mismatch error before embedding anything', async () => {
        const project = await createProject({ 'file.ts': 'const x = 1;' });

        const vectorDatabase = createVectorDatabase();
        vectorDatabase.hasCollection.mockResolvedValue(true);
        vectorDatabase.getCollectionDimension.mockResolvedValue(1536);

        const embedding = new TestEmbedding();
        const embedBatchSpy = jest.spyOn(embedding, 'embedBatch');
        const context = new Context({
            embedding,
            vectorDatabase,
            codeSplitter: new TwoChunkSplitter(),
        });

        await expect(context.indexCodebase(project)).rejects.toThrow(
            `Existing index for ${project} uses 1536-dimension vectors but the current embedding model produces 3; rebuild the index (force re-index) to switch models.`
        );

        expect(embedBatchSpy).not.toHaveBeenCalled();
        expect(vectorDatabase.query).not.toHaveBeenCalled();
        expect(vectorDatabase.insert).not.toHaveBeenCalled();
        expect(vectorDatabase.delete).not.toHaveBeenCalled();
    });

    it('fresh collection: behaves like today, with no existence-lookup queries', async () => {
        const project = await createProject({
            'one.ts': 'const one = 1;',
            'two.ts': 'const two = 2;',
        });

        const vectorDatabase = createVectorDatabase();
        vectorDatabase.hasCollection.mockResolvedValue(false);

        const embedding = new TestEmbedding();
        const embedBatchSpy = jest.spyOn(embedding, 'embedBatch');
        const context = new Context({
            embedding,
            vectorDatabase,
            codeSplitter: new TwoChunkSplitter(),
        });

        const stats = await context.indexCodebase(project);

        expect(stats.indexedFiles).toBe(2);
        expect(stats.status).toBe('completed');
        expect(vectorDatabase.query).not.toHaveBeenCalled();
        expect(vectorDatabase.getCollectionDimension).not.toHaveBeenCalled();
        expect(embedBatchSpy).toHaveBeenCalled();

        const insertedDocuments = vectorDatabase.insert.mock.calls.flatMap(([, documents]) => documents);
        expect(insertedDocuments).toHaveLength(4);
    });
});
