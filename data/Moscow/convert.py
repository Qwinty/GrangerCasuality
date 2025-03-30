import csv
import io

# --- START OF MOCK FILE CONTENT (Simulating Moscow_dtp.csv) ---
csv_data = """Показатели;"январь 2013";"февраль 2013";"март 2013";"апрель 2013";"май 2013";"июнь 2013";"июль 2013";"август 2013";"сентябрь 2013";"октябрь 2013";"ноябрь 2013";"декабрь 2013";"январь 2014";"февраль 2014";"март 2014";"апрель 2014";"май 2014";"июнь 2014";"июль 2014";"август 2014";"сентябрь 2014";"октябрь 2014";"ноябрь 2014";"декабрь 2014";"январь 2015";"февраль 2015";"март 2015";"апрель 2015";"май 2015";"июнь 2015";"июль 2015";"август 2015";"сентябрь 2015";"октябрь 2015";"ноябрь 2015";"декабрь 2015";"январь 2016";"февраль 2016";"март 2016";"апрель 2016";"май 2016";"июнь 2016";"июль 2016";"август 2016";"сентябрь 2016";"октябрь 2016";"ноябрь 2016";"декабрь 2016";"январь 2017";"февраль 2017";"март 2017";"апрель 2017";"май 2017";"июнь 2017";"июль 2017";"август 2017";"сентябрь 2017";"октябрь 2017";"ноябрь 2017";"декабрь 2017";"январь 2018";"февраль 2018";"март 2018";"апрель 2018";"май 2018";"июнь 2018";"июль 2018";"август 2018";"сентябрь 2018";"октябрь 2018";"ноябрь 2018";"декабрь 2018";"январь 2019";"февраль 2019";"март 2019";"апрель 2019";"май 2019";"июнь 2019";"июль 2019";"август 2019";"сентябрь 2019";"октябрь 2019";"ноябрь 2019";"декабрь 2019";"январь 2020";"февраль 2020";"март 2020";"апрель 2020";"май 2020";"июнь 2020";"июль 2020";"август 2020";"сентябрь 2020";"октябрь 2020";"ноябрь 2020";"декабрь 2020";"январь 2021";"февраль 2021";"март 2021";"апрель 2021";"май 2021";"июнь 2021";"июль 2021";"август 2021";"сентябрь 2021";"октябрь 2021";"ноябрь 2021";"декабрь 2021";"январь 2022";"февраль 2022";"март 2022";"апрель 2022";"май 2022";"июнь 2022";"июль 2022";"август 2022";"сентябрь 2022";"октябрь 2022";"ноябрь 2022";"декабрь 2022";"январь 2023";"февраль 2023";"март 2023";"апрель 2023";"май 2023";"июнь 2023";"июль 2023";"август 2023";"сентябрь 2023";"октябрь 2023";"ноябрь 2023";"декабрь 2023";"январь 2024";"февраль 2024";"март 2024";"апрель 2024";"май 2024";"июнь 2024";"июль 2024";"август 2024";"сентябрь 2024";"октябрь 2024"
ДТП;700;1424;2181;3043;4074;5162;6140;7214;8286;9370;10370;11319;806;1495;2329;3305;4301;5223;6280;7261;8381;9463;10332;11312;724;1430;2161;2954;3923;4802;5641;6614;7590;8501;9381;10396;657;1316;1984;2690;3460;4163;4912;5769;6726;7507;8271;9045;568;1132;1830;2483;3196;3939;4660;5476;6294;7185;8036;8907;615;1119;1797;2444;3291;4064;4786;5575;6458;7361;8229;9157;602;1224;1903;2537;3285;4040;4742;5523;6394;7318;8232;9296;728;1473;2178;2398;2728;3299;3991;4768;5628;6424;7177;7986;450;982;1659;2343;3108;3809;4569;5356;6141;6947;7732;8516;537;1038;1682;2321;3015;3738;4423;5109;5783;6494;7112;7710;499;1018;1560;2279;3005;3621;4396;5250;6128;6979;7604;8118;432;934;1434;2218;3041;3904;4832;5604;6559;7477
Погибло;47;103;151;215;282;334;416;482;574;659;746;841;52;101;160;223;305;382;471;553;637;723;798;888;41;92;131;191;254;313;374;433;497;559;612;673;34;79;122;177;224;258;303;350;395;453;514;561;28;68;103;131;170;208;257;304;346;400;439;494;40;55;88;125;161;196;240;284;332;363;409;465;33;62;97;120;154;187;231;271;309;348;389;443;26;58;84;109;134;159;193;216;262;297;327;376;18;42;62;81;109;139;167;205;243;283;323;358;18;42;65;95;118;147;175;200;221;245;277;301;17;42;55;89;115;135;168;191;216;252;283;307;13;32;59;83;119;146;183;217;245;285
Ранено;836;1663;2541;3524;4708;5952;7144;8399;9616;10828;11916;12951;950;1709;2668;3744;4898;5982;7176;8305;9540;10750;11711;12770;865;1674;2514;3389;4497;5516;6514;7642;8727;9727;10738;11903;800;1591;2365;3134;4023;4836;5703;6642;7710;8599;9467;10326;645;1291;2074;2828;3644;4495;5322;6259;7203;8186;9156;10168;710;1302;2076;2804;3773;4678;5495;6394;7418;8456;9416;10469;689;1412;2232;2950;3785;4692;5519;6456;7465;8506;9531;10723;857;1692;2506;2745;3121;3743;4553;5428;6375;7255;8111;8990;521;1112;1891;2646;3524;4321;5194;6113;6976;7900;8767;9650;652;1220;1958;2665;3440;4267;5030;5826;6598;7413;8106;8810;571;1161;1782;2550;3331;4006;4877;5821;6811;7738;8436;9000;480;1030;1604;2470;3370;4321;5337;6185;7246;8266
"""
# --- END OF MOCK FILE CONTENT ---

input_filename = 'Moscow_dtp.csv'
output_filename = 'Moscow_dtp_transformed.csv'

# Dictionary to map Russian month names to numbers
month_map = {
    "январь": "01", "февраль": "02", "март": "03", "апрель": "04",
    "май": "05", "июнь": "06", "июль": "07", "август": "08",
    "сентябрь": "09", "октябрь": "10", "ноябрь": "11", "декабрь": "12"
}

def format_date(date_str):
    """Converts 'месяц год' string to 'MM.YYYY' format."""
    try:
        # Remove potential quotes and split
        parts = date_str.strip('"').split()
        if len(parts) == 2:
            month_name, year = parts
            month_num = month_map.get(month_name.lower())
            if month_num:
                return f"{month_num}.{year}"
    except Exception as e:
        print(f"Error formatting date '{date_str}': {e}")
    return date_str # Return original if formatting fails

# Read the data
# Use io.StringIO to read the mock data string as if it were a file
# In a real scenario, you would use:
# with open(input_filename, 'r', encoding='utf-8', newline='') as infile:
with io.StringIO(csv_data) as infile:
    reader = csv.reader(infile, delimiter=';')
    all_rows = list(reader)

# Separate header and data rows
header_row = all_rows[0]
data_rows = all_rows[1:]

# Prepare the transformed data structure
transformed_data = []

# Create the new header row
# First column header is the original first cell ('Показатели')
# Subsequent column headers are the indicator names from the first column of data rows
new_header = [header_row[0]] + [row[0] for row in data_rows]
transformed_data.append(new_header)

# Get the original date headers (starting from the second column)
original_dates = header_row[1:]
num_dates = len(original_dates)
num_indicators = len(data_rows)

# Iterate through each original date column to create new rows
for i in range(num_dates):
    # Format the date for the first column of the new row
    formatted_dt = format_date(original_dates[i])
    new_row = [formatted_dt]

    # Get the corresponding data for this date from each indicator row
    for j in range(num_indicators):
        # Check if the row has enough columns for the current date index
        if i + 1 < len(data_rows[j]):
            new_row.append(data_rows[j][i+1])
        else:
            new_row.append('') # Append empty string if data is missing

    transformed_data.append(new_row)

# Write the transformed data to a new CSV file
try:
    with open(output_filename, 'w', encoding='utf-8', newline='') as outfile:
        writer = csv.writer(outfile, delimiter=';') # Using semicolon as delimiter
        writer.writerows(transformed_data)
    print(f"Successfully transformed data written to {output_filename}")

    # Optional: Print the first few rows of the transformed data
    print("\n--- First 5 rows of transformed data: ---")
    for row in transformed_data[:5]:
        print(";".join(row))

except IOError as e:
    print(f"Error writing file {output_filename}: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")