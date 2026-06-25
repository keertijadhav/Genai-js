import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { createRetriever } from "./retriver";
import { RunnableSequence } from "@langchain/core/runnables";
import { formatDocumentsAsString } from "@langchain/classic/util/document";
import { ChatHandler, chat } from "../utils/chat";
import dotenv from "dotenv";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";

dotenv.config();

const prompt = ChatPromptTemplate.fromMessages([
  [
    "human",
    `You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:`,
  ],
]);

const llm = new ChatGoogleGenerativeAI({
  model: "gemini-2.5-flash",
});

const outputParser = new StringOutputParser();
const retriever = await createRetriever();

const retrieverChain = RunnableSequence.from([
  (input) => input.question,
  retriever,
  formatDocumentsAsString,
]);

const generatorChain = RunnableSequence.from([
  {
    question: (input) => input.question,
    context: retrieverChain,
  },
  prompt,
  llm,
  outputParser,
]);

const chatHandler: ChatHandler = async (question: string) => {
  return {
    answer: generatorChain.stream({ question }),
  };
};

chat(chatHandler);