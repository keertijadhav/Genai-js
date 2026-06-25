import dotenv from "dotenv";
import { loadDocuments } from "./loadDocuments";
import { splitDocuments } from "./splitDocuments";
import { Pinecone } from "@pinecone-database/pinecone";
import cliProgress from "cli-progress";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
dotenv.config();

async function ensurePineconeIndex(pinecone: Pinecone, indexName: string, dimension: number) {
  const candidateName = `${indexName}-${dimension}`;

  try {
    const existingIndex = await pinecone.describeIndex(candidateName);
    if ((existingIndex as { dimension?: number }).dimension === dimension) {
      return candidateName;
    }
  } catch {
    // Ignore and create a new index if needed.
  }

  try {
    const existingIndex = await pinecone.describeIndex(indexName);
    if ((existingIndex as { dimension?: number }).dimension === dimension) {
      return indexName;
    }
  } catch {
    // Ignore and create a new index if needed.
  }

  console.log(`Creating Pinecone index ${candidateName} with dimension ${dimension}...`);
  await pinecone.createIndex({
    name: candidateName,
    dimension,
    spec: {
      serverless: {
        cloud: "aws",
        region: "us-east-1",
      },
    },
  });

  return candidateName;
}

const rowDocument = await loadDocuments();
const chunkedDocument = await splitDocuments(rowDocument);
const enbeddingLLM = new GoogleGenerativeAIEmbeddings({
  model: "gemini-embedding-001",
});

const pinecone = new Pinecone();
const indexName = await ensurePineconeIndex(pinecone, "langchain-docs", 3072);
const pinecodeIndex = pinecone.index(indexName);

console.log(`Starting document vectorization...`);

const progressBar = new cliProgress.SingleBar({}, cliProgress.Presets.shades_classic);

progressBar.start(chunkedDocument.length, 0);

for (let i = 0; i < chunkedDocument.length; i += 1) {
  const doc = chunkedDocument[i];
  const vector = await enbeddingLLM.embedQuery(doc.pageContent);

  if (!Array.isArray(vector) || vector.length !== 3072) {
    throw new Error(`Unexpected embedding size for chunk ${i}: ${Array.isArray(vector) ? vector.length : typeof vector}`);
  }

  await pinecodeIndex.namespace("").upsert([
    {
      id: `${Date.now()}-${i}`,
      values: vector,
      metadata: { text: doc.pageContent, source: doc.metadata?.source ?? "unknown" },
    },
  ]);
  progressBar.increment(1);
}
progressBar.stop();
console.log(`Document vectorization completed. Total chunks processed: ${chunkedDocument.length}`);