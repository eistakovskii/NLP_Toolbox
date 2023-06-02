import fitz

from pdf2image import convert_from_path
import pytesseract

import docx

import chardet
from bs4 import BeautifulSoup

import pandas as pd

from tika import parser

def extract_text_from_image(file_path: str):
    """
    The function extracts first 300 tokens from an image
    INPUT 
    file_path: path to the image file
    OUTPUT
    output_string: a normalized string without newline characters and extra spaces
    """
    pytesseract.pytesseract.tesseract_cmd = r'C:\Users\eistakovskiy\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
    out_text = pytesseract.image_to_string(file_path, lang='rus+eng')

    out_text = ' '.join(out_text.replace('\n', ' ').split())

    output_string_l = out_text.split()

    if len(output_string_l) > 300:
        output_string = ' '.join(output_string_l[:300])
    else:
        output_string = ' '.join(output_string_l)

    return output_string

def extract_text_from_pdf(file_path: str):
    """
    The function extracts first 300 tokens from a pdf document
    INPUT 
    file_path: path to the pdf file
    OUTPUT
    output_string: a normalized string without newline characters and extra spaces
    """
    
    pytesseract.pytesseract.tesseract_cmd = r'C:\Users\eistakovskiy\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
    
    pages = convert_from_path(file_path, poppler_path = r'C:\work\FILE_CLASSIFICATION\utils_in_dev\poppler_files\poppler-23.01.0\Library\bin')
    out_string_list = list()
    
    for page in pages:
        curr_string = pytesseract.image_to_string(page, lang='rus+eng')
        curr_string = ' '.join(curr_string.replace('\n', ' ').split())
        out_string_list.append(curr_string)
    
    out_text = ' '.join(out_string_list)

    output_string_l = out_text.split()

    if len(output_string_l) > 300:
        output_string = ' '.join(output_string_l[:300])
    else:
        output_string = ' '.join(output_string_l)

    return output_string

def extract_text_from_docx(file_path: str):
    """
    The function extracts first 300 tokens from a docx document
    INPUT 
    file_path: path to the docx file
    OUTPUT
    output_string: a normalized string without newline characters and extra spaces
    """
    
    doc = docx.Document(file_path)
    out_string_list = []
    
    for para in doc.paragraphs:
        out_string_list.append(para.text)
    
    out_text = ' '.join(out_string_list)
    
    out_text = ' '.join(out_text.replace('\n', ' ').split())

    output_string_l = out_text.split()

    if len(output_string_l) > 300:
        output_string = ' '.join(output_string_l[:300])
    else:
        output_string = ' '.join(output_string_l)

    return output_string

def extract_text_from_htm(file_path: str):
    """
    The function extracts first 300 tokens from a htm document
    INPUT 
    file_path: path to the htm file
    OUTPUT
    output_string: a normalized string without newline characters and extra spaces
    """
    with open(file_path, "rb") as f:
        result = chardet.detect(f.read())
    with open(file_path, "r",  encoding=result['encoding']) as f:
        soup = BeautifulSoup(f.read(), 'html.parser')
        text = soup.get_text()

    output_string_l = text.replace('\n', ' ').split()

    if len(output_string_l) > 300:
        output_string = ' '.join(output_string_l[:300])
    else:
        output_string = ' '.join(output_string_l)

    return output_string

def extract_text_from_xlsx(file_path: str):
    """
    The function extracts first 300 tokens from a xlsx document
    INPUT 
    file_path: path to the xlsx file
    OUTPUT
    output_string: a normalized string without newline characters and extra spaces
    """
    
    df = pd.DataFrame(pd.read_excel(file_path))
    df = df.fillna(0)

    total_list = list()

    cols = list(df.columns.values)
    cols = [i for i in cols if 'nnam' not in i]

    def norm_str(x):
        return ' '.join(x.replace('\n', ' ').split())

    cols = list(map(norm_str, cols))

    total_list.append(' '.join(cols))

    for i, row in df.iterrows():
        if i != 49:
            temp_l = [str(i) for i in list(row) if i != 0]
            temp_l = list(map(norm_str, temp_l))
            if len(temp_l) != 0:
                total_list.append(' '.join(temp_l)) 
        else:
            break
            
    output_string = ' '.join(total_list)

    output_string_l = output_string.split()

    if len(output_string_l) > 300:
        output_string = ' '.join(output_string_l[:300])
    else:
        output_string = ' '.join(output_string_l)

    return output_string


def extract_text_tika(input_filepath: str):

    """
    The function extracts first 300 tokens from an array of document types using library tika
    Among supported documents are doc, docs, xls, xlsx, txt
    INPUT 
    file_path: path to the file
    OUTPUT
    output_string: a normalized string without newline characters and extra spaces
    """
    
    plain_text = parser.from_file(input_filepath)
    plain_text = plain_text['content']

    clean_str = plain_text.replace('\n', ' ').replace('\t', ' ')

    output_string_l = clean_str.split()

    if len(output_string_l) > 300:
        output_string = ' '.join(output_string_l[:300])
    else:
        output_string = ' '.join(output_string_l)
    
    return output_string
  
def extract_from_txt(file_path: str):
    """
    The function extracts first 300 tokens from a txt document
    INPUT 
    file_path: path to the file
    OUTPUT
    output_string: a normalized string without newline characters and extra spaces
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    text = re.sub(r'\s+', ' ', text).strip()

    tokens = text.split(' ')

    if len(tokens) > 300:
        output_string = ' '.join(tokens[:300])
    else:
        output_string = ' '.join(tokens)
    
    return output_string
