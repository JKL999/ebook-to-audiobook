import PyPDF2
import pytesseract

# Use PyPDF2 to open the PDF file and read its contents
with open('from-third-world-to-first-by-lee-kuan-yew.pdf', 'rb') as file:
  pdf = PyPDF2.PdfReader(file)
  pages = []
  for page in range(len(pdf.pages)):
    pages.append(pdf.pages[page].extract_text())

# Concatenate all the pages to get the entire text
text = '\n'.join(pages)

# Open a new file to write the extracted text
with open('extracted_text.txt', 'w') as file:
  file.write(text)

