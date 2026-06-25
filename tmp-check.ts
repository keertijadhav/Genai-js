import { loadDocuments } from "./src/rag/loadDocuments.ts";
import { splitDocuments } from "./src/rag/splitDocuments.ts";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import dotenv from "dotenv";
dotenv.config();
const docs = await loadDocuments();
const chunks = await splitDocuments(docs);
const empty = chunks.filter((d) => !d.pageContent || !String(d.pageContent).trim());
console.log("total chunks", chunks.length);
console.log("empty chunks", empty.length);
if (empty.length) {
  console.log(empty[0]);
}
const emb = new GoogleGenerativeAIEmbeddings({ model: "gemini-embedding-001", apiKey: process.env.GOOGLE_API_KEY });
const sample = chunks.find((d) => d.pageContent && String(d.pageContent).trim())?.pageContent?.slice(0, 200);
if (sample) {
  const res = await emb.embedQuery(sample);
  console.log("sample len", res.length);
}
const validChunks = chunks.filter((d) => d.pageContent && String(d.pageContent).trim());
const batch = validChunks.slice(0, 3);
const batchRes = await emb.embedDocuments(batch.map((d) => d.pageContent));
console.log("batch lengths", batchRes.map((r) => r.length));
