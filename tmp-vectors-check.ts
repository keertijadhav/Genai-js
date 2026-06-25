import { loadDocuments } from "./src/rag/loadDocuments.ts";
import { splitDocuments } from "./src/rag/splitDocuments.ts";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import dotenv from "dotenv";
dotenv.config();
const docs = await loadDocuments();
const chunks = await splitDocuments(docs);
const emb = new GoogleGenerativeAIEmbeddings({ model: "gemini-embedding-001", apiKey: process.env.GOOGLE_API_KEY });
const batch = chunks.slice(0, 3);
const texts = batch.map((doc) => doc.pageContent);
const vectors = await emb.embedDocuments(texts);
for (let index = 0; index < batch.length; index++) {
  const vector = vectors[index];
  console.log(index, Array.isArray(vector), vector?.length, vector?.slice?.(0, 5));
}
