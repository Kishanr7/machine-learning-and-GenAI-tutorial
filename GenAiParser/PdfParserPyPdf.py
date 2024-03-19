import PyPDF2
filename = "D:\Code\Machine-Learning\GithubRepo\machine-learning-and-GenAI-tutorial\GenAiParser\multiplication-chart.pdf"
pdf_file = open(filename, 'rb')

reader = PyPDF2.PdfReader(pdf_file)

page_length = 1
page_num = 1
for i in range(page_length-1):
    page = reader.pages[i]
    text = page.extract_text()

    print('--------------------------------------------------')
    print(text)

pdf_file.close()