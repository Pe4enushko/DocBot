import logging, requests, constants, PyPDF2, os
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

AIClient = openai.OpenAI(api_key=constants.openaiAPI, timeout=None)

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

        for path in themedFiles: 
            logging.log(level=logging.INFO, msg="Starting " + os.path.basename(path) + " file processing")
            messages = [{"role": "system",
                        "content": "I will be sending a pdf file as multiple messages with text chunks. Summarize it so you will be able to give a detailed answer like a professional engineer if anyone asks"}]
            #prepare pdf file
            for chunk in pdfworker.process_pdf_to_chunks(path):
                shortened = nltkworker.shorten_technical_text(chunk)
                messages.append({"role": "system", "content": shortened})

            logging.log(level=logging.INFO, msg="Sending " + os.path.basename(path) + " to gpt")

            response = AIClient.chat.completions.create(
                messages=messages,
                model="gpt-4o-mini",
                temperature=0.6,  # Lower temperature for more deterministic output
            )


            #save summ of pdf file
            pdf_filename = os.path.basename(path)
            text_filename = os.path.splitext(pdf_filename)[0] + '.txt'

            logging.log(level=logging.INFO, msg=pdf_filename + " processing has ended")

            with open('./data/' + dirname + '/' + text_filename, 'w', encoding='utf-8') as f:
                f.write(response.choices[0].message.content)

async def getDirs(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.bot.send_message(update.effective_chat.id, text=datadirs)

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
