import pandas as pd
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import cm

# Create dummy data with drug test results
data = {
    'name': ['Pulan', 'Devi', 'Ahmad'],
    'gender': ['Laki-Laki', 'Perempuan', 'Laki-Laki'],
    'birth_date': ['1978-10-15', '1985-05-20', '1990-03-10'],
    'birth_place': ['Palu', 'Jakarta', 'Surabaya'],
    'address': ['Jalan Setia Kawan No. 5 Palu', 'Jalan Mawar No. 10 Jakarta', 'Jalan Pahlawan No. 3 Surabaya'],
    'nik': ['123456.0101978.0004', '123457.0505985.0003', '123458.0303990.0002'],
    'exam_date': ['2024-10-08', '2024-10-09', '2024-10-10'],
    'exam_result': ['Negatif', 'Positif', 'Negatif'],
    'td_nilai': ['110/70', '120/80', '115/75'],
    'nadi': ['82', '78', '80'],
    'nafas': ['18', '20', '19'],
    'amphetamine': ['Negatif', 'Negatif', 'Negatif'],
    'methamphetamine': ['Negatif', 'Positif', 'Negatif'],
    'thc': ['Negatif', 'Negatif', 'Negatif'],
    'morphine': ['Negatif', 'Negatif', 'Negatif'],
    'cocaine': ['Negatif', 'Negatif', 'Negatif'],
    'benzodiazepine': ['Negatif', 'Negatif', 'Negatif'],
    'td_id': ['3160777', '3160778', '3160779']
}

df = pd.DataFrame(data)

# Add approach to load from a csv file
df = pd.read_csv('drug_test_results.csv')

# Template for console printing
template = """
Pada hari ini {day}, tanggal {date}, telah dilakukan pemeriksaan Urine Narkoba dalam rangka
penelitian Surat Keterangan Bebas Narkoba :

A. Dasar Pemeriksaan :
   Dasar             : Informasi Lowongan Kerja PT. Langit Biru
   Tujuan            : Melamar Pekerjaan

B. Identitas         :
   Nama              : {name}
   Jenis Kelamin     : {gender}
   Tempat/Tanggal Lahir: {birth_place} / {birth_date}
   Alamat            : {address}

C. Wawancara        : Riwayat Penyalahgunaan Positif / Negatif    Positif Tanggal: {exam_date}

D. Pemeriksaan Fisik TD : {td_id} mmHg, Nadi: {nadi} X / Menit, Nafas: {nafas} X / Menit

E. NIK               : {nik}

F. Pemeriksaan Urine Narkoba menggunakan Rapid Test (Enam Parameter) dengan hasil :
   1. Amphetamine      : Positif / {amphetamine}
   2. Methamphetamine  : Positif / {methamphetamine}
   3. THC              : Positif / {thc}
   4. Morphine         : Positif / {morphine}
   5. Cocaine          : Positif / {cocaine}
   6. Benzodiazepine   : Positif / {benzodiazepine}

G. Kesimpulan        : Pada saat diperiksa {conclusion} tanda - tanda menggunakan narkoba

H. Catatan Riwayat Pemeriksaan :
   Positif
   Tanggal           : {exam_date}
   Institusi Pemeriksa: Biddokkes Polda Sulteng

                                                                    Palu, {current_date}
                                                                    Dokter Pemeriksa
"""

# Function to fill the form with all the data per row (for console printing)
def fill_form(row):
    # Convert date string to datetime for proper formatting
    exam_date = pd.to_datetime(row['exam_date'])
    birth_date = pd.to_datetime(row['birth_date'])
    
    # Get day name in Indonesian
    days_indo = {
        'Monday': 'Senin',
        'Tuesday': 'Selasa',
        'Wednesday': 'Rabu',
        'Thursday': 'Kamis',
        'Friday': 'Jumat',
        'Saturday': 'Sabtu',
        'Sunday': 'Minggu'
    }
    day_name = days_indo[exam_date.strftime('%A')]
    
    # Format dates
    formatted_exam_date = exam_date.strftime('%d - %m - %Y')
    formatted_birth_date = birth_date.strftime('%d-%m-%Y')
    current_date = datetime.now().strftime('%d October %Y')
    
    # Set conclusion based on exam result
    conclusion = "Tidak Ditemukan / Bebas" if row['exam_result'] == "Negatif" else "Ditemukan"
    
    # Fill the template
    filled_form = template.format(
        day=day_name,
        date=formatted_exam_date,
        name=row['name'],
        gender=row['gender'],
        birth_place=row['birth_place'],
        birth_date=formatted_birth_date,
        address=row['address'],
        nik=row['nik'],
        td_id=row['td_id'],
        nadi=row['nadi'],
        nafas=row['nafas'],
        amphetamine=row['amphetamine'],
        methamphetamine=row['methamphetamine'],
        thc=row['thc'],
        morphine=row['morphine'],
        cocaine=row['cocaine'],
        benzodiazepine=row['benzodiazepine'],
        exam_date=formatted_exam_date,
        conclusion=conclusion,
        current_date=current_date
    )
    
    return filled_form

# Function to create a PDF document from the filled form
def create_pdf(row):
    """Create a PDF document from the filled form that matches the example image exactly"""
    # Convert date string to datetime for proper formatting
    exam_date = pd.to_datetime(row['exam_date'])
    birth_date = pd.to_datetime(row['birth_date'])
    
    # Get day name in Indonesian
    days_indo = {
        'Monday': 'Senin',
        'Tuesday': 'Selasa',
        'Wednesday': 'Rabu',
        'Thursday': 'Kamis',
        'Friday': 'Jumat',
        'Saturday': 'Sabtu',
        'Sunday': 'Minggu'
    }
    day_name = days_indo[exam_date.strftime('%A')]
    
    # Format dates
    formatted_exam_date = exam_date.strftime('%d - %m - %Y')
    formatted_birth_date = birth_date.strftime('%d-%m-%Y')
    current_date = datetime.now().strftime('%d October %Y')
    
    # Create PDF
    filename = f"form_{row['name']}.pdf"
    doc = SimpleDocTemplate(
        filename,
        pagesize=A4,
        rightMargin=1.5*cm,
        leftMargin=1.5*cm,
        topMargin=1.5*cm,
        bottomMargin=1.5*cm
    )
    
    # Create styles
    styles = getSampleStyleSheet()
    
    # Modify existing styles instead of adding new ones
    styles['Normal'].fontName = 'Helvetica'
    styles['Normal'].fontSize = 10
    styles['Normal'].leading = 14
    
    # Create a custom header style
    styles.add(ParagraphStyle(
        name='CustomHeader',
        alignment=TA_CENTER,
        fontSize=12,
        fontName='Helvetica-Bold'
    ))
    
    # Create a right-aligned style
    styles.add(ParagraphStyle(
        name='RightAligned',
        alignment=2,  # TA_RIGHT
        fontName='Helvetica',
        fontSize=10,
        leading=14
    ))
    
    # Build PDF content
    content = []
    
    # Introduction
    intro_text = f"""Pada hari ini <b>{day_name}</b>, tanggal <b>{formatted_exam_date}</b>, telah dilakukan pemeriksaan Urine Narkoba dalam rangka penelitian Surat Keterangan Bebas Narkoba :"""
    content.append(Paragraph(intro_text, styles['Normal']))
    content.append(Spacer(1, 0.2 * cm))
    
    # Section A
    section_a = f"""<b>A. Dasar Pemeriksaan :</b><br/>
   Dasar             : Informasi Lowongan Kerja PT. Langit Biru<br/>
   Tujuan            : Melamar Pekerjaan"""
    content.append(Paragraph(section_a, styles['Normal']))
    content.append(Spacer(1, 0.3 * cm))
    
    # Section B
    section_b = f"""<b>B. Identitas :</b><br/>
   Nama              : {row['name']}<br/>
   Jenis Kelamin     : {row['gender']}<br/>
   Tempat/Tanggal Lahir: {row['birth_place']} / {formatted_birth_date}<br/>
   Alamat            : {row['address']}"""
    content.append(Paragraph(section_b, styles['Normal']))
    content.append(Spacer(1, 0.3 * cm))
    
    # Section C - Show wawancara with result
    wawancara_result = "Positif" if row['exam_result'] == "Positif" else "Negatif"
    section_c = f"""<b>C. Wawancara        :</b> Riwayat Penyalahgunaan Positif / <u>Negatif</u>    {'Positif Tanggal: ' + formatted_exam_date if wawancara_result == 'Positif' else ''}"""
    content.append(Paragraph(section_c, styles['Normal']))
    content.append(Spacer(1, 0.3 * cm))
    
    # Section D - Pemeriksaan Fisik
    section_d = f"""<b>D. Pemeriksaan Fisik TD :</b> {row['td_id']} mmHg,  Nadi: {row['nadi']} X / Menit,  Nafas: {row['nafas']} X / Menit"""
    content.append(Paragraph(section_d, styles['Normal']))
    content.append(Spacer(1, 0.3 * cm))
    
    # Section E - NIK
    section_e = f"""<b>E. NIK               :</b> {row['nik']}"""
    content.append(Paragraph(section_e, styles['Normal']))
    content.append(Spacer(1, 0.3 * cm))
    
    # Section F - Pemeriksaan Urine with drug test results
    section_f = f"""<b>F. Pemeriksaan Urine Narkoba menggunakan Rapid Test (Enam Parameter) dengan hasil :</b><br/>
   1. Amphetamine      : Positif / <u>{row['amphetamine']}</u><br/>
   2. Methamphetamine  : Positif / <u>{row['methamphetamine']}</u><br/>
   3. THC              : Positif / <u>{row['thc']}</u><br/>
   4. Morphine         : Positif / <u>{row['morphine']}</u><br/>
   5. Cocaine          : Positif / <u>{row['cocaine']}</u><br/>
   6. Benzodiazepine   : Positif / <u>{row['benzodiazepine']}</u>"""
    content.append(Paragraph(section_f, styles['Normal']))
    content.append(Spacer(1, 0.3 * cm))
    
    # Section G - Kesimpulan
    conclusion = "Tidak Ditemukan / Bebas" if row['exam_result'] == "Negatif" else "Ditemukan"
    section_g = f"""<b>G. Kesimpulan :</b> Pada saat diperiksa <b>{conclusion}</b> tanda - tanda menggunakan narkoba"""
    content.append(Paragraph(section_g, styles['Normal']))
    content.append(Spacer(1, 0.3 * cm))
    
    # Section H - Catatan Riwayat
    section_h = f"""<b>H. Catatan Riwayat Pemeriksaan :</b><br/>
   Positif<br/>
   Tanggal           : {formatted_exam_date}<br/>
   Institusi Pemeriksa: Biddokkes Polda Sulteng"""
    content.append(Paragraph(section_h, styles['Normal']))
    content.append(Spacer(1, 1.5 * cm))
    
    # Signature
    signature = f"""<para align="right">Palu, {current_date}<br/><br/><br/><br/>Dokter Pemeriksa</para>"""
    content.append(Paragraph(signature, styles['Normal']))
    
    # Build PDF
    doc.build(content)
    return filename

# Select which option to use (change this to switch between options)
options = ["Print in console", "Save to file"]

# Specify the user input for the option
user_input = input("Enter your choice (Print in console or Save to file): ")

# Print the form in console
if user_input.lower() == "print in console":
    for index, row in df.iterrows():
        filled_form = fill_form(row)
        print(f"Form for {row['name']}:")
        print(filled_form)
        print("\n" + "="*80 + "\n")

# Save the form to a PDF file
elif user_input.lower() == "save to file":
    for index, row in df.iterrows():
        pdf_file = create_pdf(row)
        print(f"PDF created: {pdf_file}")
