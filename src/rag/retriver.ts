import { VectorStoreRetriever } from "@langchain/core/vectorstores";
import { splitDocuments } from "./splitDocuments";
import { Pinecone } from "@pinecone-database/pinecone";
import cliProgress from "cli-progress";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { PineconeStore } from "@langchain/pinecone";
import dotenv from "dotenv";

dotenv.config();

export async function createRetriever(): Promise<VectorStoreRetriever> {
    const embeddingLLM = new GoogleGenerativeAIEmbeddings({
        model: "gemini-embedding-001",
    });

    const pinecone = new Pinecone();
    const pineconeIndex = pinecone.index("langchain-docs-3072");
    const VectorStore = await PineconeStore.fromExistingIndex(embeddingLLM, { pineconeIndex });

    return VectorStore.asRetriever();
}

//test case
// const retriever = await createRetriever();
// const contexts = await retriever.invoke("What is vector embedding?");
// console.log("Retriever created successfully:", retriever);
// console.log("Retrieved contexts:", contexts);