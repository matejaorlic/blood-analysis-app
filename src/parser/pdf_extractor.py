import pdfplumber

from regex_parser import extract_all_parameters


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

    # STEP 1 → PDF → raw text
    extracted_text = extract_text_from_pdf(pdf_path)

    print("\nRAW EXTRACTED TEXT:\n")
    print(extracted_text)

    # STEP 2 → raw text → structured parameters
    parsed_results = extract_all_parameters(extracted_text)

    print("\nPARSED PARAMETERS:\n")
    print(parsed_results)