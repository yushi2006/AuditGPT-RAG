import os

import streamlit as st
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain

# Import document loaders
from langchain.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredExcelLoader,
)
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma

# Import LanchChain components
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# --- SETUP ---
# تحميل متغيرات البيئة (مثل مفتاح OpenAI API)
load_dotenv()

# --- HELPER FUNCTIONS ---


def process_documents(uploaded_files):
    """
    Processes uploaded documents by loading, splitting, and embedding them.
    يعالج المستندات المرفوعة عن طريق تحميلها وتقسيمها وتحويلها إلى متجهات
    """
    all_chunks = []
    temp_dir = "temp_docs"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    for uploaded_file in uploaded_files:
        # حفظ الملف مؤقتاً لقراءته
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # اختيار الـ Loader المناسب بناءً على امتداد الملف
        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(temp_path)
        elif uploaded_file.name.endswith(".docx"):
            loader = Docx2txtLoader(temp_path)
        elif uploaded_file.name.endswith(".xlsx"):
            # ملاحظة: UnstructuredExcelLoader قد لا يكون مثالياً للجداول المعقدة
            loader = UnstructuredExcelLoader(temp_path, mode="elements")
        elif uploaded_file.name.endswith(".txt"):
            loader = TextLoader(temp_path)
        else:
            # تخطي الملفات غير المدعومة مع إشعار
            st.warning(f"File type for '{uploaded_file.name}' not supported. Skipping.")
            continue

        # تحميل وتقسيم المستند
        documents = loader.load_and_split()
        all_chunks.extend(documents)

        # حذف الملف المؤقت بعد المعالجة
        os.remove(temp_path)

    if not all_chunks:
        return None

    # إنشاء المتجهات (Embeddings) وتخزينها في ChromaDB
    embeddings = OpenAIEmbeddings(api_key=os.getenv("API_KEY"))
    vectorstore = Chroma.from_documents(documents=all_chunks, embedding=embeddings)

    return vectorstore


def get_conversation_chain(vectorstore):
    """
    Creates and returns a conversational retrieval chain.
    ينشئ ويعيد سلسلة محادثة استرجاعية
    """
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)

    # استخدام ذاكرة لحفظ سياق المحادثة
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory
    )
    return conversation_chain


# --- STREAMLIT UI ---


def main():
    st.set_page_config(
        page_title="🤖 روبوت التدقيق الداخلي", page_icon=":guardsman:", layout="wide"
    )

    # تهيئة حالة الجلسة (Session State) لحفظ المتغيرات
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "documents_processed" not in st.session_state:
        st.session_state.documents_processed = False

    # --- Sidebar for Document Upload ---
    with st.sidebar:
        st.header("مساعد التدقيق الذكي")
        st.write("ارفع مستنداتك (PDF, Word, Excel, TXT) لبدء المحادثة.")

        uploaded_files = st.file_uploader(
            "اختر ملفات التدقيق",
            type=["pdf", "docx", "xlsx", "txt"],
            accept_multiple_files=True,
        )

        if st.button("معالجة المستندات"):
            if uploaded_files:
                with st.spinner("⏳ جارٍ معالجة المستندات... قد يستغرق هذا بعض الوقت"):
                    # 1. معالجة المستندات وإنشاء قاعدة البيانات المتجهية
                    vectorstore = process_documents(uploaded_files)

                    if vectorstore:
                        # 2. إنشاء سلسلة المحادثة
                        st.session_state.conversation = get_conversation_chain(
                            vectorstore
                        )
                        st.session_state.documents_processed = True
                        st.success(
                            "✅ تم معالجة المستندات بنجاح! يمكنك الآن طرح الأسئلة."
                        )
                    else:
                        st.error(
                            "لم يتم العثور على محتوى قابل للمعالجة في الملفات المرفوعة."
                        )
            else:
                st.warning("الرجاء رفع ملف واحد على الأقل.")

    # --- Main Chat Interface ---
    st.header("🤖 روبوت التدقيق الداخلي")
    st.write("أهلاً بك! أنا هنا لمساعدتك في العثور على المعلومات بسرعة داخل مستنداتك.")
    st.markdown("---")

    # عرض سجل المحادثة
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # مربع إدخال المستخدم
    user_question = st.chat_input("اسأل عن السياسات، الإجراءات، أو اطلب تلخيصاً...")

    if user_question:
        # التأكد من أن المستندات تمت معالجتها
        if not st.session_state.documents_processed:
            st.warning("الرجاء رفع ومعالجة المستندات أولاً من الشريط الجانبي.")
            st.stop()

        # إضافة سؤال المستخدم إلى سجل المحادثة وعرضه
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        # الحصول على إجابة الروبوت
        with st.spinner("🤔 أفكر..."):
            response = st.session_state.conversation({"question": user_question})
            bot_response = response["chat_history"][
                -1
            ].content  # آخر رسالة هي إجابة الروبوت

            # إضافة إجابة الروبوت إلى السجل وعرضها
            st.session_state.chat_history.append(
                {"role": "assistant", "content": bot_response}
            )
            with st.chat_message("assistant"):
                st.markdown(bot_response)


if __name__ == "__main__":
    main()
