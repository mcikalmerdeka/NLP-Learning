import pandas as pd
from datetime import datetime

# Create dummy data
data = {
    'name': ['Pulan', 'Devi', 'Ahmad'],
    'gender': ['Laki-Laki', 'Perempuan', 'Laki-Laki'],
    'birth_date': ['1978-10-15', '1985-05-20', '1990-03-10'],
    'birth_place': ['Palu', 'Jakarta', 'Surabaya'],
    'address': ['Jalan Setia Kawan No. 5 Palu', 'Jalan Mawar No. 10 Jakarta', 'Jalan Pahlawan No. 3 Surabaya'],
    'nik': ['123456.0101978.0004', '123457.0505985.0003', '123458.0303990.0002'],
    'exam_date': ['2024-10-08', '2024-10-09', '2024-10-10'],
    'exam_result': ['Negatif', 'Positif', 'Negatif']
}

df = pd.DataFrame(data)

# Template for the document
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

D. Pemeriksaan Fisik : TD : 110 / 70 mmHg, Nadi : 82 X / Menit, Nafas : 18 X / Menit

E. NIK               : {nik}

F. Kesimpulan        : Pada saat diperiksa {exam_result} ditemukan tanda-tanda menggunakan narkoba

G. Catatan Riwayat Pemeriksaan :
   Positif
   Tanggal           : {exam_date}
   Institusi Pemeriksa: Biddokkes Polda Sulteng

                                                                    Palu, {current_date}
                                                                    Dokter Pemeriksa
"""

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
        exam_date=formatted_exam_date,
        exam_result=row['exam_result'],
        current_date=current_date
    )
    
    return filled_form

# Example usage
for index, row in df.iterrows():
    filled_form = fill_form(row)
    print(f"Form for {row['name']}:")
    print(filled_form)
    print("\n" + "="*80 + "\n")