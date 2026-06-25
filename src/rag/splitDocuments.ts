import { Document } from "@langchain/core/documents";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { loadDocuments } from "./loadDocuments";





export async function splitDocuments(rowDocuments: Document[]): Promise<Document[]> {
    console.log(`Starting document splitting...`);
    const splitter = RecursiveCharacterTextSplitter.fromLanguage("html", {
        chunkSize: 500,
        chunkOverlap: 100,
    });
    const documentChunks = await splitter.splitDocuments(rowDocuments);

    console.log(`${rowDocuments.length} documents split into ${documentChunks.length} chunks.`);

    return documentChunks;
}

// const rowDocuments = await loadDocuments();
// await splitDocuments(rowDocuments);