import { Pinecone } from "@pinecone-database/pinecone";
import dotenv from "dotenv";
dotenv.config();
const pinecone = new Pinecone();
const indexName = "langchain-docs-3072";
const index = pinecone.index(indexName);
console.log(typeof index, Object.keys(index));
console.log(typeof index.namespace, index.namespace?.toString?.());
