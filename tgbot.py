import logging, requests, constants, PyPDF2, os, math
from pathlib import Path
import time
import pdfworker, nltkworker, ollamaworker

import openai 
from telegram import Update
from telegram.ext import ApplicationBuilder, Application, CommandHandler, ContextTypes, MessageHandler, filters

datadirs = []
pdfpaths = pdfworker.find_pdf_files()
processedFiles = pdfworker.find_pdf_files('./data/')

#import nest_asyncio

#nest_asyncio.apply()

logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)

logging.getLogger("httpx").setLevel(logging.WARNING)

AIClient = openai.OpenAI(api_key=constants.openaiAPI, timeout=30)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="OwO")

async def kitty(update: Update, context: ContextTypes.DEFAULT_TYPE):
    imgurl = requests.get("https://api.thecatapi.com/v1/images/search?limit=1")
    logging.log(msg=imgurl.json()[0]['url'], level=logging.INFO)
    await context.bot.send_message(chat_id=update.effective_chat.id, text=imgurl.json()[0]['url'])


async def ai(update: Update, context: ContextTypes.DEFAULT_TYPE):
    answer = ollama_single_question(update.message.text)
    # answer = ask_openai_about_documents(get_directory_listing_compact(), update.message.text)
    await context.bot.send_message(chat_id=update.effective_chat.id, text=answer)

def ask_openai_about_documents(directory_listing: str, user_question: str, directory_path: str = "data") -> str:
    """
    Uses OpenAI GPT-4o-mini to find relevant documents and answer user question.
    
    Args:
        directory_listing (str): String containing the directory listing
        user_question (str): User's question
        directory_path (str): Base directory path for reading files
    
    Returns:
        str: Answer to the user's question
    """
    try:
        # Step 1: Ask which documents might contain the answer
        document_selection_prompt = f"""
        Given this directory structure:
        {directory_listing}
        
        And this user question: "{user_question}"
        
        Which specific file names (including their paths if in subdirectories) are most likely to contain 
        information relevant to answering this question? 
        
        Please respond with ONLY the file names (one per line) that you recommend reading to answer the question.
        If no files seem relevant, respond with "None".
        """
        
        print("🔍 Analyzing directory structure to find relevant documents...")
        
        document_response = AIClient.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that identifies relevant documents based on file names and directory structure."},
                {"role": "user", "content": document_selection_prompt}
            ],
            max_tokens=500,
            temperature=0.1
        )
        
        recommended_files_text = document_response.choices[0].message.content.strip()
        
        print(recommended_files_text)

        if recommended_files_text.lower() == "none":
            print("I couldn't find any documents that seem relevant to your question based on the file names.")
            return "I couldn't find any documents that seem relevant to your question based on the file names."
        
        # Parse the recommended files
        recommended_files = []
        for line in recommended_files_text.split('\n'):
            line = line.strip()
            if line and not line.lower().startswith(('i recommend', 'the following', 'you should')):
                # Clean up the file names and handle paths
                file_name = line.split('/')[-1]  # Get just the filename if path is given
                if file_name and '.' in file_name:  # Basic check if it looks like a file
                    recommended_files.append(file_name.strip())
        
        if not recommended_files:
            print("No specific files were recommended for your question.")
            return "No specific files were recommended for your question."
        
        print(f"📁 Recommended files: {recommended_files}")
        
        # Step 2: Read the content of recommended files
        print("📖 Reading recommended documents...")
        file_contents = {}
        base_path = Path(directory_path)
        
        for file_name in recommended_files:
            file_found = False
            
            # Search in root directory
            root_file = base_path / file_name
            if root_file.exists() and root_file.is_file():
                try:
                    with open(root_file, 'r', encoding='utf-8') as f:
                        file_contents[file_name] = f.read()
                    file_found = True
                    print(f"   ✓ Found: {file_name}")
                except Exception as e:
                    print(f"   ✗ Error reading {file_name}: {e}")
            
            # Search in subdirectories if not found in root
            if not file_found:
                for subdir in base_path.iterdir():
                    if subdir.is_dir():
                        subdir_file = subdir / file_name
                        if subdir_file.exists() and subdir_file.is_file():
                            try:
                                with open(subdir_file, 'r', encoding='utf-8') as f:
                                    file_contents[file_name] = f.read()
                                file_found = True
                                print(f"   ✓ Found: {subdir.name}/{file_name}")
                                break
                            except Exception as e:
                                print(f"   ✗ Error reading {subdir.name}/{file_name}: {e}")
            
            if not file_found:
                print(f"   ✗ Not found: {file_name}")
        
        if not file_contents:
            return "None of the recommended files could be found or read."
        
        # Step 3: Ask the question with the file contents as context
        print("🤔 Analyzing documents to answer your question...")
        
        # Prepare context from file contents
        context_parts = []
        for file_name, content in file_contents.items():
            # Limit content length to avoid token limits
            truncated_content = content[:8000] + "..." if len(content) > 8000 else content
            context_parts.append(f"--- {file_name} ---\n{truncated_content}")
        
        context = "\n\n".join(context_parts)
        
        answer_prompt = f"""
        Based on the following documents, please answer the user's question.
        
        DOCUMENTS:
        {context}
        
        USER QUESTION: {user_question}
        
        Please provide a comprehensive answer based on the document contents. 
        If the answer cannot be found in the documents, please state that clearly.
        """
        
        answer_response = AIClient.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on provided document contents."},
                {"role": "user", "content": answer_prompt}
            ],
            max_tokens=1000,
            temperature=0.3
        )
        
        final_answer = answer_response.choices[0].message.content.strip()
        
        # Add source information
        source_files = list(file_contents.keys())
        final_answer += f"\n\n📚 Sources: {', '.join(source_files)}"
        
        return final_answer
        
    except Exception as e:
        return f"Error processing your request: {str(e)}"

    
async def clearContext(update: Update, context: ContextTypes.DEFAULT_TYPE):
  context.user_data['conversation'] = None


async def setup(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pdfworker.create_directories()

def measure_tokens(messages):
    sum = 0
    for message in messages:
        sum += nltkworker.count_tokens(message["content"])
    return sum

def get_directory_listing_compact(directory_path=""):
    """
    Compact version that only shows subdirectories and their files.
    """
    output_string = ""
    path = Path(directory_path)

    # Get only directories and their files
    for directory in sorted(path.iterdir()):
        if directory.is_dir():
            output_string += f"{directory.name}\n"
            # List files in each directory
            for file_item in sorted(directory.iterdir()):
                if file_item.is_file():
                    output_string += f"    {file_item.name}\n"
            output_string += "\n"
        elif directory.is_file():
            output_string += f"    {directory.name}\n"

    # Remove the last extra newline
    output_string = output_string.rstrip()
        
    return output_string

async def getDocs(update: Update, context: ContextTypes.DEFAULT_TYPE):
    dirname = ' '.join(context.args)
    logging.log(level=logging.INFO, msg=dirname)

    if dirname in datadirs or dirname == "":
        logging.log(level=logging.INFO, msg="Searching for docs in " + "data/" + dirname)
        files = get_directory_listing_compact("data/" + dirname)
        if len(files) > 0:
            await context.bot.send_message(update.effective_chat.id, text=files)
        else:
            await context.bot.send_message(update.effective_chat.id, text="No files")
    else:
        await context.bot.send_message(update.effective_chat.id, text="There is no such directory")

async def reprocess(update: Update, context: ContextTypes.DEFAULT_TYPE):
    dirname = ' '.join(context.args)
    if dirname in datadirs:
        toProcess = pdfworker.get_files_to_process()
        themedFiles = list(filter(lambda a: dirname in a, toProcess))
        logging.log(level=logging.INFO, msg=themedFiles)
        logging.log(level=logging.INFO, msg=str(len(themedFiles)) + " of files to process")

        initPrompt = {"role": "system",
                        "content": "I will be sending a pdf file as multiple messages with text chunks. Summarize it so you will be able to give a detailed answer like a professional engineer if anyone asks. You should summarize it in 8000 tokens"}
        initPromptTokenSum = nltkworker.count_tokens(initPrompt["content"])


        for path in themedFiles: 
            docname = os.path.basename(path)
            await context.bot.send_message(update.effective_chat.id, text="working on " + docname + "...")
            logging.log(level=logging.INFO, msg="Starting " + docname + " file processing")

            messages = list()
            messages.append(initPrompt)
            #prepare pdf file
            tokenSum = initPromptTokenSum
            doneChunks = 0
            finalDoc = ""
            chunks = pdfworker.process_pdf_to_chunks(path, 9500)
            
            logging.log(level=logging.INFO, msg="count of chunks: " + str(len(chunks)))
            
            percentMessage = await context.bot.send_message(update.effective_chat.id, text="0%")


            for chunk in chunks:
                shortened = nltkworker.shorten_technical_text(chunk)
                
                tokenCount = nltkworker.count_tokens(shortened)

                if (tokenSum + tokenCount >= 40000 or doneChunks + 1 == len(chunks)):
                    logging.log(level=logging.INFO, msg="Talking to AI...")
                    await context.bot.edit_message_text(text="Talking to AI...", message_id=percentMessage.message_id, chat_id=percentMessage.chat_id)

                    response = AIClient.chat.completions.create(
                        messages=messages,
                        model="gpt-4o-mini",
                        temperature=0.6,
                        max_tokens=8000  # Lower temperature for more deterministic output
                    )

                    finalDoc = response.choices[0].message.content

                    messages.clear()
                    messages.append(initPrompt)
                    messages.append({"role": "system", "content": "Next message is the content you have already summarized. Rest ones are remaining content to summarize. Continue working on it"})
                    messages.append({"role": "system", "content": finalDoc})
                    
                    logging.log(level=logging.INFO, msg=messages)

                    tokenSum = nltkworker.count_tokens(messages[0]["content"] + messages[1]["content"] + messages[2]["content"])

                    logging.log(level=logging.INFO, msg=response.choices[0].message)
                    
                    sleepmsg = await context.bot.send_message(update.effective_chat.id, text="sleeping for some time cuz of limit of 6ok tokens per minute (but it blocks for more bruh)")
                    time.sleep(300) # 60k tokens is 1 min limit so we sleep it
                    await context.bot.delete_message(chat_id=sleepmsg.chat_id, message_id=sleepmsg.message_id)

                    #################### test bullshit
                    pdf_filename = os.path.basename(path)
                    text_filename = os.path.splitext(pdf_filename)[0] + '.txt'
                    with open('./data/' + dirname + '/' + text_filename, 'w', encoding='utf-8') as f:
                        f.write("UNFINISHED DOC " + finalDoc)

                else:
                    tokenSum += tokenCount
                    messages.append({"role": "system", "content": shortened})

                tmpmsg = await context.bot.send_message(update.effective_chat.id, text=str(tokenSum))
                time.sleep(1)
                await context.bot.delete_message(chat_id=tmpmsg.chat_id, message_id=tmpmsg.message_id)

                doneChunks += 1
                
                await context.bot.edit_message_text(text=str(doneChunks / len(chunks) * 100) + "%", message_id=percentMessage.message_id, chat_id=percentMessage.chat_id)

            #save summ of pdf file
            pdf_filename = os.path.basename(path)
            text_filename = os.path.splitext(pdf_filename)[0] + '.txt'

            logging.log(level=logging.INFO, msg=pdf_filename + " processing has ended")

            with open('./data/' + dirname + '/' + text_filename, 'w', encoding='utf-8') as f:
                f.write(finalDoc)
            await context.bot.send_message(update.effective_chat.id, text="done with this one")

def ollama_single_question(question, directory_path = "data"):        
    answer = ollamaworker.ask_ollama_qa(question, directory_path)
    return answer

async def getDirs(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(update.effective_chat.id, text="\n".join(datadirs))

if __name__ == '__main__':
    application = ApplicationBuilder().token(constants.telegramToken).build()
    start_handler = CommandHandler('start', start)
    kitty_handler = CommandHandler('kitty', kitty)
    clear_handler = CommandHandler('clear', clearContext)
    setup_handler = CommandHandler('setup', setup)
    reprocess_handler = CommandHandler('reprocess', reprocess)
    dirs_handler = CommandHandler('dirs', getDirs)
    docs_handler = CommandHandler('docs', getDocs)
    message_handler = MessageHandler(filters.TEXT & ~filters.COMMAND, ai)

    application.add_handlers([start_handler, kitty_handler, message_handler, setup_handler, reprocess_handler, dirs_handler, docs_handler])

    ollamaworker.setup_ollama_qa('/home/pe4enushko/Documents/Literature/')

    datadirs = pdfworker.get_data_dirs()

    application.run_polling()
