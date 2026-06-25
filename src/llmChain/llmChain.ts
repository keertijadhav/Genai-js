import {PromptTemplate} from "@langchain/core/prompts"
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import dotenv from "dotenv";
import { StringOutputParser } from "@langchain/core/output_parsers";
import {LLMChain} from "@langchain/classic/chains";
import { RunnableSequence } from "@langchain/core/runnables";


dotenv.config();
await personalisedPitch("Generative AI", "Javascript Developer", 100);

async function personalisedPitch(
    course: string,
    role: string,
    wordLimit: number
)
{
    const promptTemplate = new PromptTemplate({
    template: "Describe the importance of learning {course} for a {role}. Limit the output to {wordLimit} words.",
    inputVariables: ["course", "role", "wordLimit"],
 });

 const formattedPrompt = await promptTemplate.format({
    course,
    role,
    wordLimit,
 });

 console.log("Formatted Prompt:", formattedPrompt);
 const llm = new ChatGoogleGenerativeAI({
     model: "gemini-2.5-flash",
    //temperature: 1,
    //topP: 1,
    maxTokens: 80,
    
 });
 const outputParser = new StringOutputParser();

 //Option1: Langchain legacy LLMChain
//  const llmChain = new LLMChain({
//     prompt: promptTemplate,
//     llm,
//     outputParser,
//  });
//  const answer = await llmChain.invoke({
//     course,
//     role,
//     wordLimit,
//  });
//  console.log("Answer from LLMChain: ", answer);

//Option2: LCEL (Langchain Expression Language)
//const lcelChain = promptTemplate.pipe(llm).pipe(outputParser);
const lcelChain =  RunnableSequence.from([promptTemplate, llm, outputParser]);
const lcelResponse = await lcelChain.invoke({
    course,
    role,
    wordLimit,
});
console.log("Answer from LCEL Chain: ", lcelResponse);
}