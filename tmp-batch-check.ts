import { loadDocuments } from "./src/rag/loadDocuments.ts";
import { splitDocuments } from "./src/rag/splitDocuments.ts";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import dotenv from "dotenv";
dotenv.config();
const docs = await loadDocuments();
const chunks = await splitDocuments(docs);
const batch = chunks.slice(100, 111);
console.log("batch size", batch.length);
for (const [idx, chunk] of batch.entries()) {
  const text = String(chunk.pageContent ?? "");
  console.log("chunk", 100 + idx, "len", text.length, "preview", JSON.stringify(text.slice(0, 120)));
  if (!text.trim()) {
    console.log("EMPTY CHUNK");
  }
}
const emb = new GoogleGenerativeAIEmbeddings({ model: "gemini-embedding-001", apiKey: process.env.GOOGLE_API_KEY });
const vectors = await emb.embedDocuments(batch.map((chunk) => String(chunk.pageContent ?? "")));
console.log("vector lengths", vectors.map((v) => v.length));
