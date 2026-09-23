import { GoogleGenAI } from '@google/genai';
import { GeminiEmbedding } from './gemini-embedding';

const mockEmbedContent = jest.fn();

jest.mock('@google/genai', () => ({
    GoogleGenAI: jest.fn().mockImplementation(() => ({
        models: {
            embedContent: mockEmbedContent
        }
    }))
}));

describe('GeminiEmbedding', () => {
    beforeEach(() => {
        mockEmbedContent.mockReset();
        (GoogleGenAI as unknown as jest.Mock).mockClear();
    });

    it('exposes Gemini Embedding 2 model metadata', () => {
        const supportedModels = GeminiEmbedding.getSupportedModels();

        expect(supportedModels['gemini-embedding-2']).toMatchObject({
            dimension: 3072,
            contextLength: 8192,
        });

        const embedding = new GeminiEmbedding({
            apiKey: 'test-api-key',
            model: 'gemini-embedding-2',
        });

        expect(embedding.getDimension()).toBe(3072);
        expect(embedding.getSupportedDimensions()).toContain(3072);
        expect(embedding.getSupportedDimensions()).toContain(768);
    });

    it('keeps batched request behavior for Gemini Embedding 2', async () => {
        mockEmbedContent.mockResolvedValue({
            embeddings: [
                { values: [1, 0, 0] },
                { values: [0, 1, 0] },
            ],
        });

        const embedding = new GeminiEmbedding({
            apiKey: 'test-api-key',
            model: 'gemini-embedding-2',
        });

        const embeddings = await embedding.embedBatch(['first chunk', 'second chunk']);

        expect(embeddings).toEqual([
            { vector: [1, 0, 0], dimension: 3 },
            { vector: [0, 1, 0], dimension: 3 },
        ]);
        expect(mockEmbedContent).toHaveBeenCalledTimes(1);
        expect(mockEmbedContent).toHaveBeenCalledWith({
            model: 'gemini-embedding-2',
            contents: ['first chunk', 'second chunk'],
            config: {
                outputDimensionality: 3072,
            },
        });
    });

    it('keeps the existing batched request behavior for Gemini Embedding 001', async () => {
        mockEmbedContent.mockResolvedValue({
            embeddings: [
                { values: [1, 0, 0] },
                { values: [0, 1, 0] },
            ],
        });

        const embedding = new GeminiEmbedding({
            apiKey: 'test-api-key',
            model: 'gemini-embedding-001',
        });

        const embeddings = await embedding.embedBatch(['first chunk', 'second chunk']);

        expect(embeddings).toEqual([
            { vector: [1, 0, 0], dimension: 3 },
            { vector: [0, 1, 0], dimension: 3 },
        ]);
        expect(mockEmbedContent).toHaveBeenCalledTimes(1);
        expect(mockEmbedContent).toHaveBeenCalledWith({
            model: 'gemini-embedding-001',
            contents: ['first chunk', 'second chunk'],
            config: {
                outputDimensionality: 3072,
            },
        });
    });

    it('throws a clear error when a batched response count does not match the inputs', async () => {
        mockEmbedContent.mockResolvedValue({
            embeddings: [
                { values: [1, 0, 0] },
            ],
        });

        const embedding = new GeminiEmbedding({
            apiKey: 'test-api-key',
            model: 'gemini-embedding-001',
        });

        await expect(embedding.embedBatch(['first chunk', 'second chunk']))
            .rejects
            .toThrow('Gemini API returned 1 embeddings for 2 inputs');
    });

    it('returns an empty batch without calling the Gemini API', async () => {
        const embedding = new GeminiEmbedding({
            apiKey: 'test-api-key',
            model: 'gemini-embedding-2',
        });

        await expect(embedding.embedBatch([])).resolves.toEqual([]);
        expect(mockEmbedContent).not.toHaveBeenCalled();
    });

    it('passes only the API key to the GoogleGenAI client for API-key configs', () => {
        new GeminiEmbedding({
            apiKey: 'test-api-key',
            model: 'gemini-embedding-001',
        });

        const constructorArgs = (GoogleGenAI as unknown as jest.Mock).mock.calls[0][0];
        expect(constructorArgs.apiKey).toBe('test-api-key');
        expect(constructorArgs).not.toHaveProperty('vertexai');
        expect(constructorArgs).not.toHaveProperty('project');
        expect(constructorArgs).not.toHaveProperty('location');
    });

    it('passes vertexai, project and location to the GoogleGenAI client without an API key', () => {
        new GeminiEmbedding({
            model: 'gemini-embedding-001',
            vertexai: true,
            project: 'test-project',
            location: 'us-central1',
        });

        const constructorArgs = (GoogleGenAI as unknown as jest.Mock).mock.calls[0][0];
        expect(constructorArgs).toEqual({
            vertexai: true,
            project: 'test-project',
            location: 'us-central1',
        });
        expect(constructorArgs).not.toHaveProperty('apiKey');
    });

    it('still maps baseURL to httpOptions.baseUrl', () => {
        new GeminiEmbedding({
            apiKey: 'test-api-key',
            model: 'gemini-embedding-001',
            baseURL: 'https://example.com/custom',
        });

        const constructorArgs = (GoogleGenAI as unknown as jest.Mock).mock.calls[0][0];
        expect(constructorArgs.httpOptions).toEqual({ baseUrl: 'https://example.com/custom' });
    });

    describe('retry behavior', () => {
        it('retries embedBatch once on a 429 status error then succeeds', async () => {
            const retryableError = Object.assign(new Error('Too Many Requests'), { status: 429 });
            mockEmbedContent
                .mockRejectedValueOnce(retryableError)
                .mockResolvedValueOnce({
                    embeddings: [
                        { values: [1, 0, 0] },
                        { values: [0, 1, 0] },
                    ],
                });

            const embedding = new GeminiEmbedding({
                apiKey: 'test-api-key',
                model: 'gemini-embedding-001',
                retryBaseDelayMs: 1,
                retryMaxDelayMs: 2,
            });

            const embeddings = await embedding.embedBatch(['first chunk', 'second chunk']);

            expect(embeddings).toEqual([
                { vector: [1, 0, 0], dimension: 3 },
                { vector: [0, 1, 0], dimension: 3 },
            ]);
            expect(mockEmbedContent).toHaveBeenCalledTimes(2);
        });

        it('rejects immediately on a 400 status error without retrying', async () => {
            const nonRetryableError = Object.assign(new Error('Bad Request'), { status: 400 });
            mockEmbedContent.mockRejectedValueOnce(nonRetryableError);

            const embedding = new GeminiEmbedding({
                apiKey: 'test-api-key',
                model: 'gemini-embedding-001',
                retryBaseDelayMs: 1,
                retryMaxDelayMs: 2,
            });

            await expect(embedding.embedBatch(['first chunk', 'second chunk']))
                .rejects
                .toThrow('Gemini batch embedding failed: Bad Request');
            expect(mockEmbedContent).toHaveBeenCalledTimes(1);
        });

        it('rejects after exhausting retries on a persistent 503 status error', async () => {
            const retryableError = Object.assign(new Error('Service Unavailable'), { status: 503 });
            mockEmbedContent.mockRejectedValue(retryableError);

            const embedding = new GeminiEmbedding({
                apiKey: 'test-api-key',
                model: 'gemini-embedding-001',
                maxRetries: 2,
                retryBaseDelayMs: 1,
                retryMaxDelayMs: 2,
            });

            await expect(embedding.embedBatch(['first chunk', 'second chunk']))
                .rejects
                .toThrow('3 attempts');
            expect(mockEmbedContent).toHaveBeenCalledTimes(3);
        });

        it('retries embedBatch once on a network fetch failure then succeeds', async () => {
            const networkError = Object.assign(new TypeError('fetch failed'), {
                cause: { code: 'ECONNRESET' },
            });
            mockEmbedContent
                .mockRejectedValueOnce(networkError)
                .mockResolvedValueOnce({
                    embeddings: [
                        { values: [1, 0, 0] },
                        { values: [0, 1, 0] },
                    ],
                });

            const embedding = new GeminiEmbedding({
                apiKey: 'test-api-key',
                model: 'gemini-embedding-001',
                retryBaseDelayMs: 1,
                retryMaxDelayMs: 2,
            });

            const embeddings = await embedding.embedBatch(['first chunk', 'second chunk']);

            expect(embeddings).toEqual([
                { vector: [1, 0, 0], dimension: 3 },
                { vector: [0, 1, 0], dimension: 3 },
            ]);
            expect(mockEmbedContent).toHaveBeenCalledTimes(2);
        });

        it('retries embedBatch once on a quoted RESOURCE_EXHAUSTED status in the message then succeeds', async () => {
            const quotedStatusError = new Error('Request failed: {"error":{"code":429,"message":"Quota exceeded","status":"RESOURCE_EXHAUSTED"}}');
            mockEmbedContent
                .mockRejectedValueOnce(quotedStatusError)
                .mockResolvedValueOnce({
                    embeddings: [
                        { values: [1, 0, 0] },
                        { values: [0, 1, 0] },
                    ],
                });

            const embedding = new GeminiEmbedding({
                apiKey: 'test-api-key',
                model: 'gemini-embedding-001',
                retryBaseDelayMs: 1,
                retryMaxDelayMs: 2,
            });

            const embeddings = await embedding.embedBatch(['first chunk', 'second chunk']);

            expect(embeddings).toEqual([
                { vector: [1, 0, 0], dimension: 3 },
                { vector: [0, 1, 0], dimension: 3 },
            ]);
            expect(mockEmbedContent).toHaveBeenCalledTimes(2);
        });

        it('retries embed() once on a 429 status error then succeeds', async () => {
            const retryableError = Object.assign(new Error('Too Many Requests'), { status: 429 });
            mockEmbedContent
                .mockRejectedValueOnce(retryableError)
                .mockResolvedValueOnce({
                    embeddings: [
                        { values: [1, 0, 0] },
                    ],
                });

            const embedding = new GeminiEmbedding({
                apiKey: 'test-api-key',
                model: 'gemini-embedding-001',
                retryBaseDelayMs: 1,
                retryMaxDelayMs: 2,
            });

            const result = await embedding.embed('a single chunk');

            expect(result).toEqual({ vector: [1, 0, 0], dimension: 3 });
            expect(mockEmbedContent).toHaveBeenCalledTimes(2);
        });

        it('uses GEMINI_MAX_RETRIES from the environment when maxRetries is not set in config', async () => {
            const previousValue = process.env.GEMINI_MAX_RETRIES;
            process.env.GEMINI_MAX_RETRIES = '1';

            try {
                const retryableError = Object.assign(new Error('Service Unavailable'), { status: 503 });
                mockEmbedContent.mockRejectedValue(retryableError);

                const embedding = new GeminiEmbedding({
                    apiKey: 'test-api-key',
                    model: 'gemini-embedding-001',
                    retryBaseDelayMs: 1,
                    retryMaxDelayMs: 2,
                });

                await expect(embedding.embedBatch(['first chunk', 'second chunk']))
                    .rejects
                    .toThrow('2 attempts');
                expect(mockEmbedContent).toHaveBeenCalledTimes(2);
            } finally {
                if (previousValue === undefined) {
                    delete process.env.GEMINI_MAX_RETRIES;
                } else {
                    process.env.GEMINI_MAX_RETRIES = previousValue;
                }
            }
        });
    });
});
