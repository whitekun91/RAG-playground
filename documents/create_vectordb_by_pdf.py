from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFium2Loader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path
import fitz  # PyMuPDF
import os
import json
import warnings

warnings.simplefilter(action='ignore', category=UserWarning)
import posthog

posthog.disabled = True

class PDFProcessor:
    """
    A class to handle PDF processing, including image extraction and vector database creation.
    """

    def __init__(self, data_path="./raw", db_path="vector_db", extracted_image_base="./extracted_images"):
        self.data_path = Path(data_path)
        self.db_path = db_path
        self.extracted_image_base = Path(extracted_image_base)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

    # ─────────────── PDF 이미지 추출 함수 ───────────────
    def extract_images_from_pdf(self, pdf_path: str, output_dir: str):
        """
        Extract images from a PDF file and save them to the specified directory.

        Args:
            pdf_path (str): Path to the PDF file.
            output_dir (str): Directory to save the extracted images.
        """
        os.makedirs(output_dir, exist_ok=True)
        doc = fitz.open(pdf_path)

        for page_index, page in enumerate(doc):
            images = page.get_images(full=True)
            for img_index, img in enumerate(images):
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)

                # CMYK/alpha 채널 → RGB 변환
                if pix.n > 4:
                    pix = fitz.Pixmap(fitz.csRGB, pix)

                image_path = os.path.join(output_dir, f"page{page_index + 1}_img{img_index + 1}.png")
                pix.save(image_path)

    # ─────────────── 문서 로딩 및 이미지 연결 ───────────────
    def process_pdf_to_vectordb(self, pdf_path: str):
        """
        Process a PDF file to create a vector database.

        Args:
            pdf_path (str): Path to the PDF file.
        """
        pdf_name = Path(pdf_path).stem

        # 이미지 추출
        image_output_dir = self.extracted_image_base / pdf_name
        self.extract_images_from_pdf(pdf_path, str(image_output_dir))

        loader = PyPDFium2Loader(pdf_path)
        pages = loader.load()

        for i, page in enumerate(pages):
            page_number = i + 1
            chunks = self.text_splitter.split_documents([page])

            # 페이지별 이미지 파일 경로 모음 (서버 URL)
            image_folder = image_output_dir
            image_refs = []
            if image_folder.exists():
                image_refs = sorted([
                    f"./extracted_images/{pdf_name}/" + img.name
                    for img in image_folder.glob(f"page{page_number}_img*.png")
                ])
            image_refs_str = json.dumps(image_refs)

            # chunk별 문서 생성
            for chunk in chunks:
                if not chunk.page_content.strip():
                    continue  # 빈 chunk는 건너뜀!
                doc = Document(
                    page_content=chunk.page_content,
                    metadata={
                        "source": str(pdf_path),
                        "page": page_number,
                        "image_refs": image_refs_str
                    }
                )

        # ─────────────── 벡터DB 저장 ───────────────
        embeddings = HuggingFaceEmbeddings(
            model_name="../models/embeddings/ko-sbert-sts",
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True}
        )

        persist_db = Chroma.from_documents(
            documents=docs_with_images,
            embedding=embeddings,
            persist_directory=self.db_path,
            collection_name="pdf_db"
        )

# Example usage:
# pdf_processor = PDFProcessor()
# pdf_processor.extract_images_from_pdf("example.pdf", "./output_images")
# pdf_processor.process_pdf_to_vectordb("example.pdf")
