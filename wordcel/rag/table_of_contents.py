"""Indexing PDFs with tables of contents."""
from PyPDF2 import PdfReader
from llama_index import Document, LLMPredictor, ServiceContext
from llama_index import ListIndex, VectorStoreIndex
from langchain.chat_models import ChatOpenAI
from ..llm_providers import openai_call

CHATGPT_KWARGS = {"temperature": 0, "model_name": "gpt-3.5-turbo"}


def init_index(docs, index_type="vector", model="gpt-3.5-turbo"):
    """Initialize each index with a different service context."""
    assert index_type in ("vector", "list")
    assert isinstance(docs, list), "`docs` param must be a list, not a string."
    docs = [Document(text=doc) if isinstance(doc, str) else doc for doc in docs]

    kwargs = CHATGPT_KWARGS.copy()
    kwargs["model"] = model
    llm = LLMPredictor(llm=ChatOpenAI(**kwargs))
    service_context = ServiceContext.from_defaults(llm_predictor=llm)
    if index_type == "vector":
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
    elif index_type == "list":
        index = ListIndex.from_documents(docs, service_context=service_context)

    return index


def _is_page_table_of_contents(page_of_text):
    """Ask language model whether the given text is part of a table of contents."""
    prompt = page_of_text
    prompt += "\n=========\n"
    prompt += "The above text is a page of text from a PDF. Can you tell me "
    prompt += "whether it is a table of contents? Respond with 'yes' or 'no'.\n\n"
    resp = openai_call(prompt, max_tokens=1)
    return resp.lower().startswith("yes")


def extract_headings_from_toc(toc_string):
    """Given a table of contents string, extract the headings."""
    prompt = toc_string
    prompt += "\n=========\n"
    prompt += "The above text is a table of contents from a PDF. Can you tell me "
    prompt += "what the headings are? Respond with a list of headings, separated "
    prompt += "by newlines.\n\n"
    resp = openai_call(prompt)
    return resp.split("\n")


def find_table_of_contents(pages_of_text):
    """Find table of contents from the text."""
    first_half_pages = pages_of_text[: len(pages_of_text) // 2]

    found_beginning = False
    toc_string = ""
    import pdb; pdb.set_trace()
    for page_of_text in first_half_pages:
        page_is_toc = _is_page_table_of_contents(page_of_text)
        if page_is_toc:
            found_beginning = True
            toc_string += page_of_text + "\n"

        # Conclude if we've already found the beginning and the page is not 
        # the table of contents.
        if found_beginning and not page_is_toc:
            return toc_string
        
    return toc_string


def load_pdf_pages(filename):
    """Load PDF pages from `filename`."""
    assert filename.endswith(".pdf"), "File must be a PDF."
    reader = PdfReader(filename)
    pages = [page for page in reader.pages]
    pages_of_text = [page.extract_text() for page in pages]
    return pages_of_text


class TableOfContentsRetrieval():
    """TableOfContentsRetrieval sets up a two-part indexing strategy for PDFs
    that have tables of contents. It routes queries through an index of the 
    table of contents first, and then finds the relevant section of the PDF
    and queries that as well.
    ."""
    def __init__(self, filename):
        self.filename = filename
        print("Loading PDF pages...")
        self.pages_of_text = load_pdf_pages(filename)
        print("Attempting to find table of contents...")
        self.table_of_contents = find_table_of_contents(self.pages_of_text)
        if self.table_of_contents:
            print("Found table of contents.")
            headings = extract_headings_from_toc(self.table_of_contents)
            import pdb; pdb.set_trace()
            self.table_of_contents_index = init_index(
                self.table_of_contents, index_type="list"
            )
        else:
            print("No table of contents found.")
            self.table_of_contents_index = None
