import { Pinecone } from "@pinecone-database/pinecone";
import dotenv from "dotenv";
dotenv.config();
const pinecone = new Pinecone();
const index = pinecone.index("langchain-docs-3072");
const vector = Array(3072).fill(0.1);
await index.namespace("").upsert([{ id: "sdk-shape-test", values: vector, metadata: { text: "hello" } }]);
console.log("ok");
