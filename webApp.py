#import Libs
import streamlit as st
import style
import streamlitconf 


import time
from PIL import Image 
# from langchain_community.document_loaders import UnstructuredPDFLoader
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
# from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_groq import ChatGroq
from langchain.text_splitter import CharacterTextSplitter
# from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader
import os 


#env
os.environ["GROQ_API_KEY"] = "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"

#llm
llm = ChatGroq(temperature=0.8, model_name="llama3-70b-8192")

#get_pdf
def get_pdf_text(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text
data = get_pdf_text('./سوالات.pdf')

#get text_chunk
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=900,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

#split
splits = get_text_chunks(data)

#get vectordb
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings()
    knowledge_base = FAISS.from_texts(text_chunks,embeddings)
    return knowledge_base
vector_db = get_vectorstore(splits)

#RAG
QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an Iranian AI language model assistant. Your task is to generate three
            different versions of the given user question to retrieve relevant documents from
            a vector database. By generating multiple perspectives on the user question, your
            goal is to help the user overcome some of the limitations of the distance-based
            similarity search. Provide these alternative questions separated by newlines.
            Original question: {question}""",
    )

retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(),
        llm,
        prompt=QUERY_PROMPT
    )

WRITER_SYSTEM_PROMPT = "You are an IRANIAN AI critical thinker research assistant. Your sole purpose is to write well written, critically acclaimed, objective and structured reports on given text."  # noqa: E501
    # Report prompts from https://github.com/assafelovic/gpt-researcher/blob/master/gpt_researcher/master/prompts.py
RESEARCH_REPORT_TEMPLATE = """Information:
    --------
#     يکي از آيتم هاي حکم حقوقي در شرکت موننکو، اياب و ذهاب مي باشد که براساس ابلاغيه گروه مپنا که مبناي آن رده هاي
# سازماني مي باشند اعمال مي گردد. رده ها از کارمند تا مدير ارشد و مبالغ آن محرمانه مي  باشد.  
# گام نخست اين است که پست سازماني مد نظر در چارت سازماني معاونت مربوطه تعريف شده باشد. سپس معاونت مربوطه فرم  
# اعالم نياز جذب را تکميل و براي معاونت منابع انساني ارسال مي نمايد. معاونت منابع انساني پس از کنترل و تکميل فرم آن را  
# براي مديريت محترم عامل ارسال کرده و پس از تاييد مديريت محترم عامل ثبت آگهي و دريافت رزومه آغاز ميگردد.  
# رزومه هاي منتخب را به معاونت منابع انساني ارسال کرده و اين رزومه ها در فراگستر ثبت و دعوت به مصاحبه ميشوند و بعد از  
# تاييد مصاحبه فني، جهت تعيين حقوق و توافق حقوق و مزايا به معاونت منابع انساني ارجاع مي شوند. معاونت منابع انساني با   
# افراد مورد نظر تماس گرفته و حقوق و مزايا ( کارانه  -پاداش  –رفاهي) را به ايشان توضيح داده  و در صورت تاييد فرد رزومه به  
# مرحله کانون ارزيابي و در صورت تاييد کانون، فرد منتخب نهايي به مديرعامل ارجاع مي گردد.  
# طبق قانون کار و قوانين داخلي گروه مپنا و موننکو، در صورتي که تاريخ عقد بعد از تاريخ استخدام  در موننکو باشد همکار مي  
# تواند از 3   روز مرخصي اضطراري– ازدواج استفاده نمايد.  لازم به ذکر مي باشد بهره مندي از مرخصي ازدواج نيز مانند ساير  
# مرخصي ها مي بايست با تاييد مدير مربوطه انجام گردد.  
# روال در متن گواهي سابقه ي کار درج آخرين پست سازماني در حکم مي باشد. چنانچه درخواست اضافه نمودن شرح شغل را   
# داشته باشيد بايد حداکثر  3   شرح وظايف داراي اولويت را پس از تاييد مدير و يا سرپرست مستقيم به معاونت منابع انساني
# ارسال نماييد تا در گواهي سابقه ي کار اعمال گردد.  
# با استعلامي که معاونت منابع انساني از معاونت مالي اخذ مي نمايد، امکان درج ميانگين ماهيانه ميسر است.  
# مطابق مقررات داخلي موننکو در صورت بروز بيماري براي همکار، مي تواند تا سقف 6 روز از مرخصي استعالجي با  ارائه ويزيت
# پزشک و اعلام نياز به استراحت در آن و ارسال آن به امور کارکنان استفاده نمايد و حقوق بيماري تا سقف 6 روز توسط موننکو
# پرداخت مي گردد. الزم به ذکر است براي استفاده از مرخصي استعالجي بيش از سقف اعلام شده، مي بايست مدارک مربوط به  
# بيماري را در قسمت مربوطه سابت تامين اجتماعي بارگذاري گردد و پس از تاييد کميسيون پزشکي حقوق ايام بيماري را دريافت  
# مي نمايند  
# بعد از تکميل پرونده استخدامي که مهم ترين آنها گواهي سابقه کار –ليست بيمه  -مدرک تحصيلي مي باشد، معاونت منابع   
# انساني نسبت به ارزيابي حقوق با توجه به سابقه کار و مدرک اقدام مي نمايد. فرم ارزيابي ابتدا به امضاء مدير امور کارکنان سپس  
# معاونت مربوطه و نماينده مدير  عامل و در نهايت با تاييد مديريت محترم عامل مي رسد و قرارداد همکار بر اساس آن صادر مي  
# گردد.  
# مطابق ابلاغيه گروه مپنا که در ابتداي هر سال انجام مي گيرد به مادراني که فرزند زير 5  سال ( يعني تا قبل از پيش دبستاني )   
# دارند هزينه اي به عنوان هزينه مهد کودک تعلق ميگيرد که در فيش حقوق اعمال ميگردد و تاريخ پايان آن 5   سالگي اولاد مي
# باشد.   معاونين در موننکو مي توانند در صورت اعتراض و يا بازنگري حقوق پرسنل خود به معاون منابع انساني درخواستي را ارسال   
# نمايند. معاونت منابع انساني با بررسي افراد مشابه در شرکت از لحاظ سابقه و تحصيلات با فرد مورد نظر،  جدول مقايسه اي را  
# تنظيم مي نمايد. در جدول مربوطه حداقل، ميانگين و حداکثر حقوق مشابهين درج ميگردد و چنانچه فرد مورد نظر از حداقل   
# جدول مقايسه اي کمتر باشد در خصوص فرد مورد نظر تصميم گيري مي شود. نهايتا تصميمي که در مورد افزايش/عدم افزايش  
# فرد گرفته ميشود به معاون درخواست کننده اطلاع رساني مي گردد.  
# بر اساس دستورالعمل کارانه و امتياز اختصاص يافته به هر معاونت، بودجه اي بر اساس حقوق کارکنان در هر معاونت نظر گرفته  
# ميشود.  شايستگي هاي نيز تدوين شده است که براي هر فرد بر اساس شايستگي ها امتياز در نظر گرفته ميشود.   
# بر اساس سياست هر معاونت  ميتواند 1.2   تا1.5 برابر بودجه هر نفر به شرط  رعايت سقف بودجه معاونت تعيين گردد.  
# مطابق دستورالعمل جا به جايي و فرم نقل و انتقال درون سازماني، ابتدا معاونت مقصد درخواست انتقال را براي معاونت منابع  
# انساني ارسال کرده در صورت تاييد معاونت  منابع انساني فرم را به معاونت مبدا و در صورت تاييد معاونت مبدا، فرم براي همکار  
# ارسال مي گردد تا رضايت خود را اعالم نمايد و سپس براي مرحله جابه جايي نهايي به معاونت منابع انساني ارسال مي گردد. در  
# صورت عدم تاييد در معاونت مبدا موضوع منتفي مي باشد.  
# به محض دريافت نامه خاتمه همکاري در امور کارکنان، فرآيند تسويه حساب همکار در فراگستر به صورت الکترونيکي آغاز مي  
# گردد در صورتي که تمام مراحل اعم از مديرگروه/ معاونت مربوطه - کتابخانه/آموزش/ فناوري اطلاعات  / وام/مرخصي و ... انجام  
# شود به معاونت منابع انساني ارسال شده و در صورت تاييد به امور مالي جهت محاسبات و اولويت پرداخت ارسال مي گردد.   
# مطابق قانون کار براي همکاراني که صاحب فرزند ميشوند (محدوديت در تعداد فرزند نيست) و بيش از  720   روز سابقه بيمه   
# تامين اجتماعي داشته باشند ، آيتم حق اوالد که معادل 3   روز حداقل حقوق وزارت کار (يا10%   حداقل حقوق) در حکم حقوقي
# ايشان اضافه مي گردد. هم چنين براي همکاراني که فرزند باالي 18   سال داشته باشند و فرزند اشتغال به تحصيل نداشته باشد و
# يا همکار  بازنشسته گردد آيتم حق اوالد از حکم ايشان حذف مي گردد.  
# در صورتي که همکار آقا باشد مي بايست مدارک شناسنامه اي و کارت ملي همسر را علاوه بر امور کارکنان به بيمه نيز ارسال  
# نمايد. در صورتي که همکار خانم  باشد و درخواست بيمه براي همسر را داشته باشد مدارک را به بيمه نيز ارسال مي نمايد که اين  
# مورد براي همکار خانم داراي هزينه مي باشد.  هم چنين فايل آشنايي دريافت هديه ازدواج تامين اجتماعي در صورت داشتن  
# 720  روز سابقه بيمه براي همکار ارسال مي گردد.  
#        --------
#     Using the above information, answer the following question or topic: "{question}" in a short manner-- \
#     The answer should focus on the answer to the question, should be well structured, informative, \
#     in depth, with facts and numbers if available and a minimum of 100 words and a maximum of 250 words.
#     You should strive to write the answer using all relevant and necessary information provided.
#     The answer should not include the question itself.
#     You can only use '.' and '،'  and '\n' and you should not write the report with markdown syntax.
#     Avoid writing long paragraphs; After a few sentences, write the continuation of the answers on the first line.
#     You MUST determine your own concrete and valid opinion based on the given information. Do NOT deter to general and meaningless conclusions.
#     You should not write the sources used in the context, and if you use them, they should not be cited at the end of any article.
#     You have to talk in Persian language. Always assume you have to spean Persian and all the words in the context must be Persian.
#     If the question was given outside the information, answer in Persian only in one sentence: "The answer to this question is not available in my knowledge."
#     Please do your best, this is very important to my career. """  # noqa: E501


prompt = ChatPromptTemplate.from_messages(
        [
            ("system", WRITER_SYSTEM_PROMPT),
            ("user", RESEARCH_REPORT_TEMPLATE),
        ]
    )

chain = (
            {"text": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

answer = chain.invoke(
        {
            "question": input
        }
    )





    

        

                
    

    



  

   
    
    

