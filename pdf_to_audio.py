import PyPDF2
import pytesseract

# Use PyPDF2 to open the PDF file and read its contents
with open('example.pdf', 'rb') as file:
    pdf = PyPDF2.PdfFileReader(file)
    pages = []
    for page in range(pdf.getNumPages()):
        pages.append(pdf.getPage(page).extractText())
