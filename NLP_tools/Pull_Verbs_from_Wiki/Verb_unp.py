def wiki_list():
    from bs4 import BeautifulSoup
    import requests
    import pandas as pd

    url = "https://de.wiktionary.org/wiki/Verzeichnis:Deutsch/Unpers%C3%B6nliche_Verben"


    data  = requests.get(url).text

    soup = BeautifulSoup(data,"html5lib")

    tables = soup.find_all('table')


    table_index = 1

    verbs_data = pd.DataFrame(columns=["Verbs", "Comments"])

    for row in tables[table_index].tbody.find_all("tr"):
        col = row.find_all("td")
        if (col != []):
            verbs = col[0].text
            comments = col[1].text
            verbs_data = verbs_data.append({"Verbs":verbs, "Comments":comments}, ignore_index=True)


    verbs_list = verbs_data['Verbs'].tolist()
    return verbs_list