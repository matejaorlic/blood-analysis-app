import pdfplumber


def extract_text_from_pdf(pdf_path):

    full_text = ""

    with pdfplumber.open(pdf_path) as pdf:

        for page in pdf.pages:

            text = page.extract_text()

            if text:
                full_text += text + "\n"

    return full_text


if __name__ == "__main__":

    pdf_path = "data/raw/NALAZ_Mirjana_Orlic.pdf"

    extracted_text = extract_text_from_pdf(pdf_path)

    print(extracted_text)