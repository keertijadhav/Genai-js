import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { createRetriever } from "./retriver";
import { RunnableSequence } from "@langchain/core/runnables";
import { formatDocumentsAsString } from "@langchain/classic/util/document";
import { ChatHandler, chat } from "../utils/chat";
import dotenv from "dotenv";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { BaseMessage, AIMessage, HumanMessage } from "@langchain/core/messages";

dotenv.config();

const prompt = ChatPromptTemplate.fromMessages([
  [
    "system",
    `You are a helpful assistant for question answering. Use the retrieved context to answer the user's question. If the context does not contain the answer, say that you do not know. Keep the answer concise, with at most three sentences.`,
  ],
  new MessagesPlaceholder("chat_history"),
  [
    "human",
    `Question: {question}
Context:
{context}`,
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
    chat_history: (input) => input.chat_history,
  },
  prompt,
  llm,
  outputParser,
]);

const qcSystemPrompt = `Given a chat history and the latest user question, rewrite the latest question into a standalone question that can be answered without the chat history. Use the chat history only to resolve pronouns or missing context. Return only the rewritten question and nothing else.`;


const qcPrompt = ChatPromptTemplate.fromMessages([
    [
        "system",
        qcSystemPrompt,
    ],
    new MessagesPlaceholder("chat_history"),
    [
        "human",
        "{question}"
    ]
]);


const qcChain = RunnableSequence.from([qcPrompt, llm, outputParser]);
const chatHistory:BaseMessage [] = [];

const chatHandler: ChatHandler = async (question: string) => {
  let contextualQuestion: string | null = null;

  if (chatHistory.length > 0) {
    contextualQuestion = (await qcChain.invoke({
      question,
      chat_history: chatHistory,
    })) as string;
    contextualQuestion = contextualQuestion.trim();
    console.log(`Contextual Question: ${contextualQuestion}`);
  }

  return {
    answer: generatorChain.stream({
      question: contextualQuestion || question,
      chat_history: chatHistory,
    }),
    answerCallBack: async (answerText: string) => {
      chatHistory.push(new HumanMessage(contextualQuestion || question));
      chatHistory.push(new AIMessage(answerText));
    },
  };
};

chat(chatHandler);