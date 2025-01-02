from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import fitz  # PyMuPDF
import re
from pdf2image import convert_from_path
import sqlite3
import os
# Specify the PDF file path
pdf_path = './data/synonym.pdf'


def convert_pdf_to_images(pdf_path, output_folder):
    pdf_document = fitz.open(pdf_path)
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap()
        output_path = f"{output_folder}/page_{page_num + 1}.png"
        pix.save(output_path)
        print(f"Saved {output_path}")

pdf_path = "./data/synonym.pdf"
output_folder = "./data/"


def preprocess_image(image_path):
    img = Image.open(image_path)
    # Convert to sgrayscale
    img = img.convert('L')
    # Apply contrast enhancement
    img = img.filter(ImageFilter.SHARPEN)
    return img

def image_to_text(img):
    text = pytesseract.image_to_string(img)
    return text

def find_index_with_substring(lst, substring):
    for index, string in enumerate(lst):
        if substring in string:
            return index
    return -1  # Return -1 if substring not found in any element
def loadIntoThesaurusDB(result):
    conn = sqlite3.connect('my_database.db')
    cursor = conn.cursor()

    # Create a table
    cursor.execute('''CREATE TABLE IF NOT EXISTS synonymslistings
                    (id INTEGER PRIMARY KEY, term TEXT, synonym Text)''')
    
    #print("result")
    #print(result)
    for r in result:
     #   print("r")
     #   print(r)
        for syn in result[r]:
      #      print("syn")
      #      print(syn)    
            cursor.execute("SELECT * FROM synonymslistings WHERE term=? and synonym=?", (r,syn))
            ret = cursor.fetchone()
            if(ret is None):
                cursor.execute("INSERT INTO synonymslistings (term, synonym) VALUES (?, ?)", (r,syn))
    conn.commit()
    cursor.close() 
    conn.close()


# Example usage
for i in range(3, 504):
    file = f"./data/page_{i}.png"    
    if(os.path.exists(file)):
        image_path = file 
        print("file")
        print(file)
        img = preprocess_image(image_path)
        text = image_to_text(img)


        import re

        # Pattern to match keywords and their associated lists including items starting with ‘1 and excluding ‘Wr
        pattern = re.compile(r'\b[ev|Kev|Ke|KEY]+: ([\w\s,\[\]\(\)\‘\d,]+)', re.DOTALL)

        matches = pattern.findall(text)

        print(matches)

        result = {}


        for match in matches:
            items = match.strip().split("\n")
            if(len(items) == 1):
                continue
            #print("match")
            #print(match)
            #print("len(match)")
            #print(len(match))
            #print("items")
            #print(items)
            
            if "‘1" in items[find_index_with_substring(items, "‘1")]:
                keyword = items[0]
                print("keyword")
                print(keyword)
                
                synonyms = items[find_index_with_substring(items, "‘1")].split(",")
                print("synonyms")
                print(synonyms)
                if keyword not in result:
                    result[keyword] = []
                for s in synonyms:
                    tmpS = s.split(' ')
                    if len(tmpS) > 1:
                        result[keyword].append(tmpS[1])
                    else:
                        result[keyword].append(s)
            if "‘i" in items[find_index_with_substring(items, "‘i")]:
                keyword = items[0]
                #print("keyword")
                #print(keyword)
                
                synonyms = items[find_index_with_substring(items, "‘i")].split(",")
                #print("synonyms")
                #print(synonyms)
                if keyword not in result:
                    result[keyword] = []
                for s in synonyms:
                    tmpS = s.split(' ')
                    if len(tmpS) > 1:
                        result[keyword].append(tmpS[1])
                    else:
                        result[keyword].append(s)

        loadIntoThesaurusDB(result)