#--------------------------------------------------------------------------
#       
#               FLAN T5 Model        
#
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
import os
os.environ["TOKENIZERS_PARALLELISM"]="false"

#--------------------------------------------------------------------------
#Import from hugging face transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

#Autotokenizer : A Tokenizer Loader that automatically picks the right tokenizer for the model you pick
# A pretrained  model loader for Sequence to Sequence Language Models (Seq2SeqLM).

#Choose a instruction-tuned model
MODEL_NAME="google/flan-t5-small"

# A lightweight version of flan-T5.
# About 80 million parameters

print(f"Flan-T5_Summarizer_Q&A_Assistant {MODEL_NAME}Model Loading ...")

#Load Tokenizer (handles text <-> tokens)
# Auto Tokenizer picks the right tokens for the model

tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME)

#Load the Sequence to Sequence Model
model=AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

#--------------------------------------------------------------------------
def Run_Flan(prompt : str,max_new_tokens : int = 128) -> str:
    #Tokenization
    #Tokanize the input prompt return; return pytorch tensors, truncate if too long
    inputs=tokenizer(prompt,return_tensors="pt",truncation=True)

    #Generation
    #Generate text from the model with light sampling for naturalness
    outputs=model.generate(
        **inputs,           #Pass tokenized inputs  (input_ids,attention_mask)
        max_new_tokens=max_new_tokens,  #How many new tokens to generate
        do_sample=True,       #Enable random sampling
        top_p=0.9,             #Nucleus sampling : only consider tokens in top 90% probability mass
        temperature=0.7          #Control randomness (lower=Safer / more deterministic) 
    )

    #Decode token IDs back into a clean string
    #Example IDs : [71,867,1234,41,1]
    # Text :       "Hello How Are You?"
    return tokenizer.decode(outputs[0],skip_special_tokens=True).strip()
#--------------------------------------------------------------------------
#
#  This Function is used for summarization
#  It creates a prompt with 4 to 6 bullet points
#
#--------------------------------------------------------------------------

def summarized_text(text : str)->str:
    #Prompt Template instructing the model to produce 4-6 bullet points
    prompt=f"Summarize the following text in 4-6 bullet points\n\n{text}"

    #Allow a slighty longer output for bullet lists
    return Run_Flan(prompt,max_new_tokens=160)

#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
# This function is used to load the contents from our local file
# and return the complete file content in 1 string
#
#--------------------------------------------------------------------------

def load_content(path : str = "context.txt")->str:
    try:
        #Read The entire file as a single string
        with open(path,"r",encoding="utf-8") as f :
            return f.read()

    except FileNotFoundError:
        return ""

#--------------------------------------------------------------------------
# The function ask flan to answer using only the given context.
# If Answer isnt present ask it to say not found
#
#--------------------------------------------------------------------------

def answer_from_context(question : str,context : str) -> str:
    if not context.strip():
        return "Context file not found or empty create context file first"
    
    #Construct a strict prompt for flan T5
    prompt=(
        "You are a helpful assistant. Answer the question only using the context.\n"
        "If the answer is not in context, reply exactly : Not Found \n\n"

        f"Context :\n{context}\n\n"
        f"Questions : {question}\nAnswer"

    )

    #Generate a concise answer grounded in provided notes
    return Run_Flan(prompt,max_new_tokens=120)
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
#   Entry Point Function
#--------------------------------------------------------------------------

def main():
    print("---------------------------------------------------------------------")
    print("-----------------Flan T5 Model-----------------------")
    print("1. Summarize the model")
    print("2. Questions and answers over local 'context.txt' ")
    print("0. Exit")
    print("---------------------------------------------------------------------")

    while True:

        choice=input("\nChoose an option (1/2/0): ").strip()

        if choice == "0":
            print("Thank You for using Flan T5 model ")
            break

        elif choice == "1":
            #collect multiple lines of text for summarization
            print("You have selected summarization option")
            print("\n Paste text to summarize. End with a blank line :")

            lines=[]
            while True:
                line = input()

                #Stop when user heats enter on an empty line
                if not line.strip():
                    break
                lines.append(line)

            #join lines into a single block of text
            text="\n".join(lines).strip()

            #If no text was provided prompt again
            if not text:
                print("Flan T5 Says : No text received")
                continue 
            
            # Run Summarization and print the result
            print("\n Summary Generated by Flan T5")
            print(summarized_text(text))
        
        elif choice == "2":
            #Load context from local file 'Context.txt'
            ctx=load_content("context.txt")

            if not ctx.strip():
                print("Missing 'context.txt'. Create it in same folder and try again ")
                continue

            #Ask a question related to provided context
            q=input("Ask a question about your context to FLAN Model : ").strip()
            if not q:
                print("No question received ")
                continue

            #Generate an answer grounded only in the context
            print("Answer from FLAN Model ")
            print(answer_from_context(q,ctx))
        
        else:
            print("Please Choose (1,2,0)")



#--------------------------------------------------------------------------
#           Starter
#--------------------------------------------------------------------------
if __name__ == "__main__":
    main()

