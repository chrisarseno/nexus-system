// Enhanced vector operations for advanced knowledge retrieval
export enum VectorSpace {
  SEMANTIC = "semantic",
  SYNTACTIC = "syntactic", 
  MULTIMODAL = "multimodal",
  KNOWLEDGE = "knowledge",
  TEMPORAL = "temporal",
  HIERARCHICAL = "hierarchical"
}

export enum ChunkingStrategy {
  FIXED_SIZE = "fixed_size",
  SEMANTIC_BOUNDARY = "semantic_boundary",
  SLIDING_WINDOW = "sliding_window", 
  HIERARCHICAL = "hierarchical",
  ADAPTIVE = "adaptive",
  CONTEXT_AWARE = "context_aware"
}

export enum RetrievalMethod {
  COSINE_SIMILARITY = "cosine_similarity",
  EUCLIDEAN_DISTANCE = "euclidean_distance",
  DOT_PRODUCT = "dot_product",
  HYBRID_SEARCH = "hybrid_search", 
  SEMANTIC_FUSION = "semantic_fusion",
  CONTEXTUAL_RANKING = "contextual_ranking"
}

export interface VectorDocument {
  docId: string;
  content: string;
  vectorEmbedding: number[]; // Simplified from numpy array
  metadata: Record<string, any>;
  chunkIndex: number;
  parentDocId?: string;
  creationTime: Date;
  lastAccessed: Date;
  accessCount: number;
  relevanceScore: number;
  semanticCluster?: number;
}

export interface SearchResult {
  document: VectorDocument;
  similarityScore: number;
  rank: number;
  explanation: string;
  fusionScores: Record<string, number>;
}

export interface ChunkingConfig {
  strategy: ChunkingStrategy;
  maxChunkSize: number;
  overlapSize: number;
  semanticThreshold?: number;
  contextWindow?: number;
}

/**
 * Advanced Vector Engine for enhanced RAG operations
 */
export class EnhancedVectorEngine {
  private documents: Map<string, VectorDocument> = new Map();
  private semanticClusters: Map<number, string[]> = new Map();
  private accessPatterns: Map<string, number[]> = new Map();

  constructor() {
    console.log('🔍 Enhanced Vector Engine initialized');
  }

  /**
   * Add document with advanced chunking
   */
  async addDocument(
    content: string, 
    metadata: Record<string, any> = {},
    chunkingConfig: ChunkingConfig = {
      strategy: ChunkingStrategy.SEMANTIC_BOUNDARY,
      maxChunkSize: 1000,
      overlapSize: 200
    }
  ): Promise<string[]> {
    const chunks = this.chunkDocument(content, chunkingConfig);
    const docIds: string[] = [];
    const parentDocId = `doc_${Date.now()}`;

    for (let i = 0; i < chunks.length; i++) {
      const chunk = chunks[i];
      const docId = `${parentDocId}_chunk_${i}`;
      
      const document: VectorDocument = {
        docId,
        content: chunk,
        vectorEmbedding: await this.generateEmbedding(chunk),
        metadata: { ...metadata, chunkIndex: i },
        chunkIndex: i,
        parentDocId,
        creationTime: new Date(),
        lastAccessed: new Date(),
        accessCount: 0,
        relevanceScore: 0,
        semanticCluster: await this.assignSemanticCluster(chunk)
      };

      this.documents.set(docId, document);
      docIds.push(docId);
    }

    console.log(`📄 Added document with ${chunks.length} chunks using ${chunkingConfig.strategy}`);
    return docIds;
  }

  /**
   * Advanced document chunking
   */
  private chunkDocument(content: string, config: ChunkingConfig): string[] {
    switch (config.strategy) {
      case ChunkingStrategy.SEMANTIC_BOUNDARY:
        return this.semanticBoundaryChunking(content, config);
      case ChunkingStrategy.SLIDING_WINDOW:
        return this.slidingWindowChunking(content, config);
      case ChunkingStrategy.HIERARCHICAL:
        return this.hierarchicalChunking(content, config);
      case ChunkingStrategy.ADAPTIVE:
        return this.adaptiveChunking(content, config);
      case ChunkingStrategy.CONTEXT_AWARE:
        return this.contextAwareChunking(content, config);
      default:
        return this.fixedSizeChunking(content, config);
    }
  }

  /**
   * Semantic boundary chunking - split at natural semantic breaks
   */
  private semanticBoundaryChunking(content: string, config: ChunkingConfig): string[] {
    const chunks: string[] = [];
    
    // Split by paragraphs first
    const paragraphs = content.split(/\n\s*\n/).filter(p => p.trim());
    
    let currentChunk = '';
    
    for (const paragraph of paragraphs) {
      const potentialChunk = currentChunk + (currentChunk ? '\n\n' : '') + paragraph;
      
      if (potentialChunk.length <= config.maxChunkSize) {
        currentChunk = potentialChunk;
      } else {
        if (currentChunk) {
          chunks.push(currentChunk.trim());
        }
        
        // If single paragraph is too large, split by sentences
        if (paragraph.length > config.maxChunkSize) {
          const sentences = this.splitBySentences(paragraph);
          let sentenceChunk = '';
          
          for (const sentence of sentences) {
            if ((sentenceChunk + sentence).length <= config.maxChunkSize) {
              sentenceChunk += (sentenceChunk ? ' ' : '') + sentence;
            } else {
              if (sentenceChunk) {
                chunks.push(sentenceChunk.trim());
              }
              sentenceChunk = sentence;
            }
          }
          
          currentChunk = sentenceChunk;
        } else {
          currentChunk = paragraph;
        }
      }
    }
    
    if (currentChunk.trim()) {
      chunks.push(currentChunk.trim());
    }
    
    return chunks;
  }

  /**
   * Sliding window chunking with overlap
   */
  private slidingWindowChunking(content: string, config: ChunkingConfig): string[] {
    const chunks: string[] = [];
    const words = content.split(/\s+/);
    const wordsPerChunk = Math.floor(config.maxChunkSize / 6); // Approximate words per chunk
    const overlapWords = Math.floor(config.overlapSize / 6);
    
    for (let i = 0; i < words.length; i += wordsPerChunk - overlapWords) {
      const chunkWords = words.slice(i, i + wordsPerChunk);
      if (chunkWords.length > 0) {
        chunks.push(chunkWords.join(' '));
      }
    }
    
    return chunks;
  }

  /**
   * Hierarchical chunking - maintains document structure
   */
  private hierarchicalChunking(content: string, config: ChunkingConfig): string[] {
    const chunks: string[] = [];
    
    // Look for headers and sections
    const lines = content.split('\n');
    let currentSection = '';
    let sectionHeader = '';
    
    for (const line of lines) {
      const trimmedLine = line.trim();
      
      // Detect headers (simple heuristic)
      if (this.isLikelyHeader(trimmedLine)) {
        if (currentSection.trim()) {
          chunks.push((sectionHeader + '\n\n' + currentSection).trim());
        }
        sectionHeader = trimmedLine;
        currentSection = '';
      } else {
        currentSection += line + '\n';
        
        // If section gets too large, chunk it
        if (currentSection.length > config.maxChunkSize) {
          const sectionChunks = this.semanticBoundaryChunking(
            sectionHeader + '\n\n' + currentSection, 
            config
          );
          chunks.push(...sectionChunks.slice(0, -1)); // Add all but last
          currentSection = sectionChunks[sectionChunks.length - 1] || '';
        }
      }
    }
    
    if (currentSection.trim()) {
      chunks.push((sectionHeader + '\n\n' + currentSection).trim());
    }
    
    return chunks.filter(chunk => chunk.length > 50); // Filter very short chunks
  }

  /**
   * Adaptive chunking - adjusts based on content complexity
   */
  private adaptiveChunking(content: string, config: ChunkingConfig): string[] {
    const complexity = this.calculateContentComplexity(content);
    
    // Adjust chunk size based on complexity
    const adaptedConfig = {
      ...config,
      maxChunkSize: Math.floor(config.maxChunkSize * (complexity > 0.7 ? 0.8 : 1.2))
    };
    
    return this.semanticBoundaryChunking(content, adaptedConfig);
  }

  /**
   * Context-aware chunking - maintains topical coherence
   */
  private contextAwareChunking(content: string, config: ChunkingConfig): string[] {
    // For now, use semantic boundary chunking with topic detection
    const baseChunks = this.semanticBoundaryChunking(content, config);
    
    // TODO: Implement topic modeling to merge/split chunks by topic
    return baseChunks;
  }

  /**
   * Fixed size chunking (fallback)
   */
  private fixedSizeChunking(content: string, config: ChunkingConfig): string[] {
    const chunks: string[] = [];
    
    for (let i = 0; i < content.length; i += config.maxChunkSize - config.overlapSize) {
      const chunk = content.slice(i, i + config.maxChunkSize);
      if (chunk.trim()) {
        chunks.push(chunk.trim());
      }
    }
    
    return chunks;
  }

  /**
   * Split text by sentences
   */
  private splitBySentences(text: string): string[] {
    return text.match(/[^.!?]+[.!?]+/g) || [text];
  }

  /**
   * Check if line is likely a header
   */
  private isLikelyHeader(line: string): boolean {
    if (line.length === 0) return false;
    
    // Common header patterns
    const headerPatterns = [
      /^#{1,6}\s+/, // Markdown headers
      /^\d+\.\s+/, // Numbered headers
      /^[A-Z\s]{5,}$/, // ALL CAPS headers
      /^.{1,80}:?\s*$/ // Short lines that might be headers
    ];
    
    return headerPatterns.some(pattern => pattern.test(line)) && 
           line.length < 100;
  }

  /**
   * Calculate content complexity
   */
  private calculateContentComplexity(content: string): number {
    const avgWordLength = content.split(/\s+/).reduce((sum, word) => sum + word.length, 0) / content.split(/\s+/).length;
    const avgSentenceLength = content.split(/[.!?]+/).filter(s => s.trim()).length;
    const uniqueWordRatio = new Set(content.toLowerCase().split(/\s+/)).size / content.split(/\s+/).length;
    
    // Normalize and combine factors
    return Math.min(1, (avgWordLength / 10 + avgSentenceLength / 50 + uniqueWordRatio) / 3);
  }

  /**
   * Generate embeddings (simplified - in production would use actual embedding model)
   */
  private async generateEmbedding(text: string): Promise<number[]> {
    // Simplified embedding generation
    const words = text.toLowerCase().split(/\s+/);
    const embedding = new Array(384).fill(0); // Typical embedding size
    
    // Simple hash-based pseudo-embedding
    for (let i = 0; i < words.length; i++) {
      const word = words[i];
      for (let j = 0; j < word.length; j++) {
        const charCode = word.charCodeAt(j);
        embedding[j % 384] += charCode * (i + 1);
      }
    }
    
    // Normalize
    const magnitude = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
    return embedding.map(val => magnitude > 0 ? val / magnitude : 0);
  }

  /**
   * Assign semantic cluster
   */
  private async assignSemanticCluster(content: string): Promise<number> {
    // Simplified clustering based on content hash
    const hash = this.simpleHash(content.toLowerCase());
    return hash % 10; // 10 clusters
  }

  /**
   * Simple hash function
   */
  private simpleHash(str: string): number {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32bit integer
    }
    return Math.abs(hash);
  }

  /**
   * Advanced semantic search with multiple retrieval methods
   */
  async search(
    query: string,
    options: {
      method?: RetrievalMethod;
      limit?: number;
      threshold?: number;
      boostRecent?: boolean;
      clusterFilter?: number[];
    } = {}
  ): Promise<SearchResult[]> {
    const {
      method = RetrievalMethod.HYBRID_SEARCH,
      limit = 10,
      threshold = 0.1,
      boostRecent = true,
      clusterFilter
    } = options;

    const queryEmbedding = await this.generateEmbedding(query);
    const results: SearchResult[] = [];

    for (const [docId, doc] of this.documents) {
      // Apply cluster filter if specified
      if (clusterFilter && doc.semanticCluster && !clusterFilter.includes(doc.semanticCluster)) {
        continue;
      }

      let similarity = 0;
      const fusionScores: Record<string, number> = {};

      // Calculate similarity based on method
      switch (method) {
        case RetrievalMethod.COSINE_SIMILARITY:
          similarity = this.cosineSimilarity(queryEmbedding, doc.vectorEmbedding);
          break;
        case RetrievalMethod.EUCLIDEAN_DISTANCE:
          similarity = 1 / (1 + this.euclideanDistance(queryEmbedding, doc.vectorEmbedding));
          break;
        case RetrievalMethod.DOT_PRODUCT:
          similarity = this.dotProduct(queryEmbedding, doc.vectorEmbedding);
          break;
        case RetrievalMethod.HYBRID_SEARCH:
          fusionScores.semantic = this.cosineSimilarity(queryEmbedding, doc.vectorEmbedding);
          fusionScores.lexical = this.lexicalSimilarity(query, doc.content);
          fusionScores.recency = boostRecent ? this.recencyScore(doc) : 0;
          similarity = fusionScores.semantic * 0.6 + fusionScores.lexical * 0.3 + fusionScores.recency * 0.1;
          break;
        case RetrievalMethod.SEMANTIC_FUSION:
          fusionScores.semantic = this.cosineSimilarity(queryEmbedding, doc.vectorEmbedding);
          fusionScores.contextual = this.contextualRelevance(query, doc);
          similarity = (fusionScores.semantic + fusionScores.contextual) / 2;
          break;
        case RetrievalMethod.CONTEXTUAL_RANKING:
          similarity = this.contextualRanking(query, doc);
          break;
      }

      if (similarity >= threshold) {
        results.push({
          document: doc,
          similarityScore: similarity,
          rank: 0, // Will be set after sorting
          explanation: this.generateExplanation(method, similarity, fusionScores),
          fusionScores
        });
      }
    }

    // Sort by similarity and assign ranks
    results.sort((a, b) => b.similarityScore - a.similarityScore);
    results.forEach((result, index) => {
      result.rank = index + 1;
      // Update access statistics
      result.document.lastAccessed = new Date();
      result.document.accessCount++;
    });

    return results.slice(0, limit);
  }

  /**
   * Calculate cosine similarity
   */
  private cosineSimilarity(vecA: number[], vecB: number[]): number {
    const dotProduct = vecA.reduce((sum, a, i) => sum + a * vecB[i], 0);
    const magnitudeA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
    const magnitudeB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));
    
    if (magnitudeA === 0 || magnitudeB === 0) return 0;
    return dotProduct / (magnitudeA * magnitudeB);
  }

  /**
   * Calculate Euclidean distance
   */
  private euclideanDistance(vecA: number[], vecB: number[]): number {
    return Math.sqrt(vecA.reduce((sum, a, i) => sum + Math.pow(a - vecB[i], 2), 0));
  }

  /**
   * Calculate dot product
   */
  private dotProduct(vecA: number[], vecB: number[]): number {
    return vecA.reduce((sum, a, i) => sum + a * vecB[i], 0);
  }

  /**
   * Calculate lexical similarity
   */
  private lexicalSimilarity(query: string, content: string): number {
    const queryWords = new Set(query.toLowerCase().split(/\s+/));
    const contentWords = new Set(content.toLowerCase().split(/\s+/));
    
    const intersection = new Set([...queryWords].filter(word => contentWords.has(word)));
    const union = new Set([...queryWords, ...contentWords]);
    
    return intersection.size / union.size;
  }

  /**
   * Calculate recency score
   */
  private recencyScore(doc: VectorDocument): number {
    const hoursSinceAccess = (Date.now() - doc.lastAccessed.getTime()) / (1000 * 60 * 60);
    return Math.max(0, 1 - hoursSinceAccess / 168); // Decay over 1 week
  }

  /**
   * Calculate contextual relevance
   */
  private contextualRelevance(query: string, doc: VectorDocument): number {
    // Simple implementation - count query terms in document
    const queryTerms = query.toLowerCase().split(/\s+/);
    const content = doc.content.toLowerCase();
    
    let matches = 0;
    let totalWeight = 0;
    
    queryTerms.forEach((term, index) => {
      const termFreq = (content.match(new RegExp(term, 'g')) || []).length;
      const weight = 1 / (index + 1); // Earlier terms weighted more
      matches += termFreq * weight;
      totalWeight += weight;
    });
    
    return Math.min(1, matches / Math.max(1, totalWeight));
  }

  /**
   * Calculate contextual ranking
   */
  private contextualRanking(query: string, doc: VectorDocument): number {
    const semanticScore = this.lexicalSimilarity(query, doc.content);
    const importanceScore = doc.relevanceScore;
    const popularityScore = Math.min(1, doc.accessCount / 100);
    
    return semanticScore * 0.5 + importanceScore * 0.3 + popularityScore * 0.2;
  }

  /**
   * Generate explanation for search result
   */
  private generateExplanation(
    method: RetrievalMethod, 
    similarity: number, 
    fusionScores: Record<string, number>
  ): string {
    let explanation = `Similarity: ${similarity.toFixed(3)} (${method})`;
    
    if (Object.keys(fusionScores).length > 0) {
      const scoreDetails = Object.entries(fusionScores)
        .map(([key, value]) => `${key}: ${value.toFixed(3)}`)
        .join(', ');
      explanation += ` | Fusion scores: ${scoreDetails}`;
    }
    
    return explanation;
  }

  /**
   * Get document statistics
   */
  getStats(): {
    totalDocuments: number;
    totalChunks: number;
    semanticClusters: number;
    avgChunkSize: number;
    avgAccessCount: number;
  } {
    const docs = Array.from(this.documents.values());
    
    return {
      totalDocuments: new Set(docs.map(d => d.parentDocId)).size,
      totalChunks: docs.length,
      semanticClusters: new Set(docs.map(d => d.semanticCluster)).size,
      avgChunkSize: docs.reduce((sum, d) => sum + d.content.length, 0) / Math.max(1, docs.length),
      avgAccessCount: docs.reduce((sum, d) => sum + d.accessCount, 0) / Math.max(1, docs.length)
    };
  }

  /**
   * Clear all documents
   */
  clear(): void {
    this.documents.clear();
    this.semanticClusters.clear();
    this.accessPatterns.clear();
    console.log('🗑️ Vector engine cleared');
  }
}