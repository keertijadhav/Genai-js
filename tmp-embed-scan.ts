import { loadDocuments } from "./src/rag/loadDocuments.ts";
import { splitDocuments } from "./src/rag/splitDocuments.ts";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import dotenv from "dotenv";
dotenv.config();
const docs = await loadDocuments();
const chunks = await splitDocuments(docs);
const emb = new GoogleGenerativeAIEmbeddings({ model: "gemini-embedding-001", apiKey: process.env.GOOGLE_API_KEY });
for (let i = 0; i < chunks.length; i++) {
  const text = String(chunks[i].pageContent ?? "");
  if (!text.trim()) {
    console.log("empty chunk", i);
    continue;
  }
  try {
    const vector = await emb.embedQuery(text);
    if (!Array.isArray(vector) || vector.length !== 3072) {
      console.log("bad vector at", i, "len", Array.isArray(vector) ? vector.length : typeof vector);
      console.log(text.slice(0, 200));
      break;
    }
  } catch (err) {
    console.log("embed error at", i, err.message);
    console.log(text.slice(0, 200));
    break;
  }
}
console.log("finished scan");
