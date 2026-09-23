import { GoogleGenAI } from '@google/genai';
import { Embedding, EmbeddingVector } from './base-embedding';
import { envManager } from '../utils/env-manager';

type GeminiModelInfo = {
    dimension: number;
    contextLength: number;
    description: string;
    supportedDimensions?: number[];
};

export interface GeminiEmbeddingConfig {
    model: string;
    apiKey?: string; // Gemini API key; omit when using Vertex AI with Application Default Credentials
    baseURL?: string; // Optional custom API endpoint URL
    outputDimensionality?: number; // Optional dimension override
    vertexai?: boolean; // Optional: call Gemini through Vertex AI instead of the Gemini Developer API
    project?: string; // Google Cloud project ID (Vertex AI)
    location?: string; // Google Cloud location, e.g. us-central1 (Vertex AI)
    maxRetries?: number; // Retries after the first attempt for transient errors (default 8, env GEMINI_MAX_RETRIES)
    retryBaseDelayMs?: number; // Initial backoff delay in ms, doubles per retry (default 1000, env GEMINI_RETRY_BASE_DELAY_MS)
    retryMaxDelayMs?: number; // Maximum backoff delay in ms (default 32000)
}

const RETRYABLE_HTTP_STATUSES = new Set([408, 429, 500, 502, 503, 504]);
const RETRYABLE_NETWORK_CODES = new Set([
    'ECONNRESET',
    'ETIMEDOUT',
    'EAI_AGAIN',
    'EPIPE',
    'ECONNABORTED',
    'UND_ERR_SOCKET',
    'UND_ERR_CONNECT_TIMEOUT',
    'UND_ERR_HEADERS_TIMEOUT',
    'UND_ERR_BODY_TIMEOUT',
]);
const RETRYABLE_QUOTED_STATUS_PATTERN = /"status"\s*:\s*"(RESOURCE_EXHAUSTED|UNAVAILABLE|DEADLINE_EXCEEDED)"/;

const DEFAULT_MAX_RETRIES = 8;
const DEFAULT_RETRY_BASE_DELAY_MS = 1000;
const DEFAULT_RETRY_MAX_DELAY_MS = 32000;

export class GeminiEmbedding extends Embedding {
    private client: GoogleGenAI;
    private config: GeminiEmbeddingConfig;
    private dimension: number = 3072; // Default dimension for Gemini embedding models
    protected maxTokens: number = 2048; // Maximum tokens for Gemini embedding models
    private maxRetries: number; // Retries after the first attempt for transient errors
    private retryBaseDelayMs: number; // Initial backoff delay in ms, doubles per retry
    private retryMaxDelayMs: number; // Maximum backoff delay in ms

    constructor(config: GeminiEmbeddingConfig) {
        super();
        this.config = config;
        this.client = new GoogleGenAI({
            ...(config.apiKey !== undefined && { apiKey: config.apiKey }),
            ...(config.vertexai !== undefined && { vertexai: config.vertexai }),
            ...(config.project && { project: config.project }),
            ...(config.location && { location: config.location }),
            ...(config.baseURL && {
                httpOptions: {
                    baseUrl: config.baseURL
                }
            }),
        });

        this.maxRetries = this.resolveRetrySetting(
            'maxRetries',
            config.maxRetries,
            'GEMINI_MAX_RETRIES',
            DEFAULT_MAX_RETRIES,
            value => Number.isInteger(value) && value >= 0
        );
        this.retryBaseDelayMs = this.resolveRetrySetting(
            'retryBaseDelayMs',
            config.retryBaseDelayMs,
            'GEMINI_RETRY_BASE_DELAY_MS',
            DEFAULT_RETRY_BASE_DELAY_MS,
            value => Number.isInteger(value) && value >= 1
        );
        this.retryMaxDelayMs = this.resolveRetrySetting(
            'retryMaxDelayMs',
            config.retryMaxDelayMs,
            undefined,
            Math.max(DEFAULT_RETRY_MAX_DELAY_MS, this.retryBaseDelayMs),
            value => Number.isInteger(value) && value >= this.retryBaseDelayMs
        );

        // Set dimension based on model and configuration
        this.updateDimensionForModel(config.model || 'gemini-embedding-001');

        // Override dimension if specified in config
        if (config.outputDimensionality) {
            this.dimension = config.outputDimensionality;
        }
    }

    private updateDimensionForModel(model: string): void {
        const supportedModels = GeminiEmbedding.getSupportedModels();
        const modelInfo = supportedModels[model];

        if (modelInfo) {
            this.dimension = modelInfo.dimension;
            this.maxTokens = modelInfo.contextLength;
        } else {
            // Use default dimension and context length for unknown models
            this.dimension = 3072;
            this.maxTokens = 2048;
        }
    }

    async detectDimension(): Promise<number> {
        // Gemini doesn't need dynamic detection, return configured dimension
        return this.dimension;
    }

    async embed(text: string): Promise<EmbeddingVector> {
        const processedText = this.preprocessText(text);
        const model = this.config.model || 'gemini-embedding-001';

        try {
            return await this.embedProcessedText(processedText, model);
        } catch (error) {
            throw new Error(this.formatFailureMessage('Gemini embedding failed', error));
        }
    }

    async embedBatch(texts: string[]): Promise<EmbeddingVector[]> {
        if (texts.length === 0) {
            return [];
        }

        const processedTexts = this.preprocessTexts(texts);
        const model = this.config.model || 'gemini-embedding-001';

        try {
            const response = await this.withRetry(() => this.client.models.embedContent({
                model: model,
                contents: processedTexts,
                config: {
                    outputDimensionality: this.config.outputDimensionality || this.dimension,
                },
            }));

            if (!response.embeddings) {
                throw new Error('Gemini API returned invalid response');
            }

            if (response.embeddings.length !== processedTexts.length) {
                throw new Error(`Gemini API returned ${response.embeddings.length} embeddings for ${processedTexts.length} inputs`);
            }

            return response.embeddings.map((embedding: any) => {
                if (!embedding.values) {
                    throw new Error('Gemini API returned invalid embedding data');
                }
                return {
                    vector: embedding.values,
                    dimension: embedding.values.length
                };
            });
        } catch (error) {
            throw new Error(this.formatFailureMessage('Gemini batch embedding failed', error));
        }
    }

    private async embedProcessedText(processedText: string, model: string): Promise<EmbeddingVector> {
        const response = await this.withRetry(() => this.client.models.embedContent({
            model: model,
            contents: processedText,
            config: {
                outputDimensionality: this.config.outputDimensionality || this.dimension,
            },
        }));

        if (!response.embeddings || !response.embeddings[0] || !response.embeddings[0].values) {
            throw new Error('Gemini API returned invalid response');
        }

        return {
            vector: response.embeddings[0].values,
            dimension: response.embeddings[0].values.length
        };
    }

    /**
     * Resolves a retry-related setting from, in order, the constructor
     * config, an environment variable (when one applies), and finally the
     * default. Config and env values share the same validation: an invalid
     * value is never silently coerced, it's logged and replaced by the
     * default so a typo can't disable retries or spin the loop forever.
     */
    private resolveRetrySetting(
        settingName: string,
        configValue: number | undefined,
        envVarName: string | undefined,
        defaultValue: number,
        isValid: (value: number) => boolean
    ): number {
        if (configValue !== undefined) {
            if (isValid(configValue)) {
                return configValue;
            }
            console.warn(`[GeminiEmbedding] ⚠️  Invalid ${settingName} (${configValue}); using default ${defaultValue}`);
            return defaultValue;
        }

        const envValue = envVarName ? envManager.get(envVarName) : undefined;
        if (envValue !== undefined) {
            const parsed = Number(envValue);
            if (isValid(parsed)) {
                return parsed;
            }
            console.warn(`[GeminiEmbedding] ⚠️  Invalid ${envVarName} (${envValue}); using default ${defaultValue}`);
            return defaultValue;
        }

        return defaultValue;
    }

    private async withRetry<T>(operation: () => Promise<T>): Promise<T> {
        // this.maxRetries is validated at construction, but this guard keeps
        // the loop from misbehaving (0 or negative attempts, or a fractional
        // attempt count) if that ever changes.
        const maxAttempts = Math.max(1, Math.floor(this.maxRetries) + 1);

        for (let attempt = 1; attempt <= maxAttempts; attempt++) {
            try {
                return await operation();
            } catch (error) {
                const retryable = this.isRetryableError(error);
                const isFinalAttempt = attempt === maxAttempts;

                if (!retryable || isFinalAttempt) {
                    if (retryable && isFinalAttempt && error && typeof error === 'object') {
                        (error as Record<string, unknown>).attempts = attempt;
                    }
                    throw error;
                }

                const delayMs = this.computeRetryDelayMs(attempt);
                console.warn(`[GeminiEmbedding] ⚠️  Embedding request failed (${this.describeError(error)}), retrying in ${Math.round(delayMs)}ms (retry ${attempt}/${this.maxRetries})`);
                await this.sleep(delayMs);
            }
        }

        throw new Error('Gemini embedding retry loop exited unexpectedly');
    }

    private isRetryableError(error: unknown): boolean {
        // @google/genai's Node transport catches a fetch rejection and
        // rethrows it as a plain Error, e.g. "exception TypeError: fetch
        // failed sending request", dropping the original TypeError and its
        // cause (so the code/cause.code branch below can't see it). Match on
        // the message instead of the error's type or a cause code.
        if (error instanceof Error && /fetch failed/i.test(error.message)) {
            return true;
        }

        if (!error || typeof error !== 'object') {
            return false;
        }

        const { status, code } = this.extractStatusAndCode(error);

        if (status !== undefined) {
            return RETRYABLE_HTTP_STATUSES.has(status);
        }

        const message = typeof (error as { message?: unknown }).message === 'string' ? (error as { message: string }).message : '';
        if (RETRYABLE_QUOTED_STATUS_PATTERN.test(message)) {
            return true;
        }

        // Still reachable from google-auth-library/gaxios token-refresh
        // errors, which do carry a code (e.g. ECONNRESET) rather than the
        // generic "fetch failed" message above.
        return code !== undefined && RETRYABLE_NETWORK_CODES.has(code);
    }

    /**
     * Best-effort HTTP status and error code extraction, shared by
     * isRetryableError and describeError: a direct property for SDK errors,
     * or a nested one (response.status, cause.code) for gaxios/
     * google-auth-library errors.
     */
    private extractStatusAndCode(error: unknown): { status?: number; code?: string } {
        if (!error || typeof error !== 'object') {
            return {};
        }

        const err = error as {
            status?: unknown;
            response?: { status?: unknown };
            code?: unknown;
            cause?: { code?: unknown };
        };

        const status = typeof err.status === 'number'
            ? err.status
            : (err.response && typeof err.response.status === 'number' ? err.response.status : undefined);
        const code = typeof err.code === 'string'
            ? err.code
            : (err.cause && typeof err.cause.code === 'string' ? err.cause.code : undefined);

        return { status, code };
    }

    private describeError(error: unknown): string {
        if (error && typeof error === 'object') {
            const { status, code } = this.extractStatusAndCode(error);
            const identifier = status !== undefined ? status : code;
            const message = typeof (error as { message?: unknown }).message === 'string' ? (error as { message: string }).message : String(error);

            return identifier !== undefined ? `${identifier}: ${message}` : message;
        }

        return String(error);
    }

    private computeRetryDelayMs(attempt: number): number {
        const cap = Math.min(this.retryMaxDelayMs, this.retryBaseDelayMs * Math.pow(2, attempt - 1));
        return cap / 2 + Math.random() * (cap / 2);
    }

    private sleep(ms: number): Promise<void> {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    private formatFailureMessage(prefix: string, error: unknown): string {
        const message = error instanceof Error ? error.message : 'Unknown error';
        const attempts = error && typeof error === 'object' ? (error as Record<string, unknown>).attempts : undefined;

        return typeof attempts === 'number'
            ? `${prefix} after ${attempts} ${attempts === 1 ? 'attempt' : 'attempts'}: ${message}`
            : `${prefix}: ${message}`;
    }

    getDimension(): number {
        return this.dimension;
    }

    getProvider(): string {
        return 'Gemini';
    }

    /**
     * Set model type
     * @param model Model name
     */
    setModel(model: string): void {
        this.config.model = model;
        this.updateDimensionForModel(model);
    }

    /**
     * Set output dimensionality
     * @param dimension Output dimension (must be supported by the model)
     */
    setOutputDimensionality(dimension: number): void {
        this.config.outputDimensionality = dimension;
        this.dimension = dimension;
    }

    /**
     * Get client instance (for advanced usage)
     */
    getClient(): GoogleGenAI {
        return this.client;
    }

    /**
     * Get list of supported models
     */
    static getSupportedModels(): Record<string, GeminiModelInfo> {
        return {
            'gemini-embedding-001': {
                dimension: 3072,
                contextLength: 2048,
                description: 'Gemini embedding model with state-of-the-art performance',
                supportedDimensions: [3072, 1536, 768, 256] // Matryoshka Representation Learning support
            },
            'gemini-embedding-2': {
                dimension: 3072,
                contextLength: 8192,
                description: 'Gemini Embedding 2 model with improved embedding quality and longer context',
                supportedDimensions: [3072, 1536, 768, 256]
            }
        };
    }

    /**
     * Get supported dimensions for the current model
     */
    getSupportedDimensions(): number[] {
        const modelInfo = GeminiEmbedding.getSupportedModels()[this.config.model || 'gemini-embedding-001'];
        return modelInfo?.supportedDimensions || [this.dimension];
    }

    /**
     * Validate if a dimension is supported by the current model
     */
    isDimensionSupported(dimension: number): boolean {
        const supportedDimensions = this.getSupportedDimensions();
        return supportedDimensions.includes(dimension);
    }
}
