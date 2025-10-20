import logging, requests, constants, PyPDF2, os, math
import time
import pdfworker, nltkworker

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
    if 'conversation' not in context.user_data:
        context.user_data['conversation'] = []

    context.user_data['conversation'].append({
        "role": "user",
        "content": update.message.text
    })

    # TODO:Implement gpt choosing which docs to open to read info and answer user's question

    response = advanced_usage(context.user_data['conversation'])
    await context.bot.send_message(chat_id=update.effective_chat.id, text=response)
    
async def clearContext(update: Update, context: ContextTypes.DEFAULT_TYPE):
  context.user_data['conversation'] = None

def advanced_usage(requests):
    """Show advanced usage with more parameters"""
    #client = openai.OpenAIClient()
    
    messages = [
        {"role": "system", "content": "you are the most cute cat femboy in the world nya"},
    ] + [
        {"role": msg['role'], "content": msg['content']} 
        for msg in requests
    ]
    
    response = AIClient.chat.completions.create(
        messages=messages,
        model="gpt-4o-mini",
        temperature=0.5,  # Lower temperature for more deterministic output
        max_tokens=500    # Limit response length
    )
    return response.choices[0].message.content

async def setup(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pdfworker.create_directories()

    
async def reprocess(update: Update, context: ContextTypes.DEFAULT_TYPE):
    dirname = ' '.join(context.args)
    if dirname in datadirs:
        toProcess = pdfworker.get_files_to_process()
        themedFiles = list(filter(lambda a: dirname in a, toProcess))
        logging.log(level=logging.INFO, msg=themedFiles)
        logging.log(level=logging.INFO, msg=str(len(themedFiles)) + " of files to process")

        initPrompt = {"role": "system",
                        "content": "I will be sending a pdf file as multiple messages with text chunks. Summarize it so you will be able to give a detailed answer like a professional engineer if anyone asks"}
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

                if (tokenSum + tokenCount >= 50000 or doneChunks + 1 == len(chunks)):
                    logging.log(level=logging.INFO, msg="Talking to AI...")
                    await context.bot.edit_message_text(text="Talking to AI...", message_id=percentMessage.message_id, chat_id=percentMessage.chat_id)

                    response = AIClient.chat.completions.create(
                        messages=messages,
                        model="gpt-4o-mini",
                        temperature=0.6,
                        max_tokens=7000  # Lower temperature for more deterministic output
                    )

                    finalDoc = response.choices[0].message.content

                    messages.clear()
                    messages.append(initPrompt)
                    messages.append({"role": "system", "content": "Next message is the content you have already summarized. Rest ones are remaining content to summarize. Continue working on it"})
                    messages.append({"role": "system", "content": finalDoc})
                    
                    logging.log(level=logging.INFO, msg=messages)

                    tokenSum = nltkworker.count_tokens(messages[0]["content"] + messages[1]["content"] + messages[2]["content"])

                    logging.log(level=logging.INFO, msg=response.choices[0].message)
                    
                    sleepmsg = await context.bot.send_message(update.effective_chat.id, text="sleeping for 60s cuz of limit of 6ok tokens per minute")
                    time.sleep(70) # 60k tokens is 1 min limit so we sleep it
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
    message_handler = MessageHandler(filters.TEXT & ~filters.COMMAND, ai)

    application.add_handlers([start_handler, kitty_handler, message_handler, setup_handler, reprocess_handler, dirs_handler])

    datadirs = pdfworker.get_data_dirs()

    application.run_polling()
