import json
import streamlit as st
from streamlit.components.v1 import html
import pandas as pd  # type: ignoreimport plotly.express as px
from pycaret.clustering import load_model, predict_model  # type: ignore
import plotly.express as px  # type: ignore

MODEL_NAME = 'welcome_survey_clustering_pipeline_v2'

DATA = 'welcome_survey_simple_v2.csv'

CLUSTER_NAMES_AND_DESCRIPTIONS = 'welcome_survey_cluster_names_and_descriptions_v2.json'


@st.cache_data
def get_model():
    return load_model(MODEL_NAME)

@st.cache_data
def get_cluster_names_and_descriptions():
    with open(CLUSTER_NAMES_AND_DESCRIPTIONS, "r", encoding='utf-8') as f:
        return json.loads(f.read())

@st.cache_data
def get_all_participants():
    all_df = pd.read_csv(DATA, sep=';')
    df_with_clusters = predict_model(model, data=all_df)

    return df_with_clusters


###########################################     SIDEBAR     ################################
#
with st.sidebar:
    st.header("Powiedz nam coś o sobie")
    st.markdown("Pomożemy Ci znaleźć osoby, które mają podobne zajafki :)")
    age = st.selectbox("Wiek", ['<18', '25-34', '45-54', '35-44', '18-24', '>=65', '55-64', 'unknown'])
    edu_level = st.selectbox("Wykształcenie", ['Podstawowe', 'Średnie', 'Wyższe'])
    fav_animals = st.selectbox("Ulubione zwierzęta", ['Brak ulubionych', 'Psy', 'Koty', 'Inne', 'Koty i Psy'])
    fav_place = st.selectbox("Ulubione miejsce", ['Nad wodą', 'W lesie', 'W górach', 'Inne'])
    gender = st.radio("Płeć", ['Mężczyzna', 'Kobieta'])

    person_df = pd.DataFrame([
        {
            'age': age,
            'edu_level': edu_level,
            'fav_animals': fav_animals,
            'fav_place': fav_place,
            'gender': gender,
        }
    ])

##############################      Edycja DF       ##################
#

model = get_model()
all_df = get_all_participants()
cluster_names_and_descriptions = get_cluster_names_and_descriptions()

predicted_cluster_id = predict_model(model, data=person_df)["Cluster"].values[0]
predicted_cluster_data = cluster_names_and_descriptions[predicted_cluster_id]

# Zmiana nazwy niepodanych wartości
all_df["age"] = all_df["age"].replace("unknown", "niepodany")

# Zmiana nazw kolumn
all_df = all_df.rename(columns={
    "age": "Wiek",
    "edu_level": "Wykształcenie",
    "fav_animals": "Ulubione_zwierze",
    "fav_place": "Ulubione_miejsce",
    "gender": "Płeć"
})

same_cluster_df = all_df[all_df["Cluster"] == predicted_cluster_id]

# Zmiana nazwy "unknown" na "niepodany" w kolumnie "age"
same_cluster_df["Wiek"] = same_cluster_df["Wiek"].replace("unknown", "niepodany")


####################################    DATE FRAME i ILOŚĆ (Ogół)
#
# Wylicz procentowy rozkład klastra
cluster_counts = all_df["Cluster"].value_counts(normalize=True) * 100  # Licz procenty
cluster_counts_df = cluster_counts.reset_index()  # Przekształć na DataFrame
cluster_counts_df.columns = ["Cluster", "Percentage"]  # Nazwij kolumny

# Pobierz nazwy klastrów z pliku JSON
cluster_names_and_descriptions = get_cluster_names_and_descriptions()

# Zmapuj identyfikatory klastrów na nazwy
cluster_counts_df["Cluster Name"] = cluster_counts_df["Cluster"].map(
    lambda x: cluster_names_and_descriptions[x]["name"]
)

allal = len(all_df)

####################################        WYKRES OGÓLNY
#

st.header(f"Liczba ankietowanych: {allal}")
all_df

##################################      WYKRES PROCENTOWY DLA GRUP
#

# Tworzenie wykresu słupkowego
fig = px.bar(
    cluster_counts_df,
    x="Cluster Name",  # Osie X - klastery
    y="Percentage",  # Osie Y - procenty
    text="Percentage",  # Dodanie wartości jako etykiet
    color="Cluster",  # Kolory na podstawie klastra
    title="Procentowy rozkład klastrów",
    color_discrete_sequence=px.colors.qualitative.Set2  # Ustawienie kolorów
)

# Dostosowanie wyglądu
fig.update_traces(
    texttemplate='%{text:.2f}%',  # Format procentów
    textposition='outside'  # Wyświetlanie tekstu nad słupkami
)
# Zwiększenie marginesów i dostosowanie osi Y
fig.update_layout(
    xaxis_title="Klaster",
    yaxis_title="Procent",
    title_font=dict(size=20, color="#B2FF59", family="Arial"),  # Styl tytułu wykresu
    font=dict(color="#B2FF59"),  # Kolor ogólny dla czcionek
    xaxis=dict(
        tickfont=dict(size=14, color="#B2FF59"),  # Czcionka osi X
        title=dict(font=dict(size=16, color="#B2FF59"))  # Kolor i rozmiar tytułu osi X
    ),
    yaxis=dict(
        tickfont=dict(size=14, color="#B2FF59"),  # Czcionka osi Y
        title=dict(font=dict(size=16, color="#B2FF59")),  # Kolor i rozmiar tytułu osi Y
        automargin=True,  # Automatyczne marginesy
        range=[0, cluster_counts_df["Percentage"].max() * 1.2]  # Zwiększenie zakresu osi Y
    ),
    margin=dict(t=70, b=50, l=50, r=50),  # Marginesy wykresu
    legend=dict(
        title=dict(font=dict(size=12, color="#B2FF59")),  # Tytuł legendy
        font=dict(size=12, color="#B2FF59")  # Czcionka elementów legendy
    )
)
# Wyświetlenie wykresu w Streamlit
st.plotly_chart(fig)




####################################    Nazwa Grupy
#

st.header("Najbliżej Ci do grupy: ")
st.subheader(f"{predicted_cluster_data['name']}")

st.markdown(predicted_cluster_data['description'])


# Stylizacja i wyświetlenie metryki w jednej linii
html(f"""
    <div style=" 
        color: #B2FF59; 
        font-family: Arial, sans-serif; 
        font-size: 16px; 
    ">
        <span style=>Liczba twoich znajomych:</span>
        <strong>{len(same_cluster_df)}</strong>
    </div>
""")

####################################    WykresY
#


# Dane dla wykresów
wiek_counts = same_cluster_df["Wiek"].value_counts().reset_index()
wiek_counts.columns = ["Wiek", "count"]  # Nazwij kolumny


# Filtruj dane, aby usunąć wartości z liczbą 0
filtered_wiek_counts = wiek_counts[wiek_counts["count"] > 0]

# Wspólne kolory dla obu wykresów
colors = px.colors.qualitative.Set2


####################################    Wykres 1
#

bar_fig = px.bar(
    filtered_wiek_counts,
    x="Wiek",  # Oś X - wiek
    y="count",  # Oś Y - liczba
    text="count",  # Wyświetlanie liczby na słupkach
    color="Wiek",  # Kolory na podstawie wieku
    color_discrete_sequence=colors  # Wspólne kolory
)
bar_fig.update_layout(
    xaxis_title="Wiek",
    yaxis_title="Liczba osób",
    title="Rozkład wieku w grupie (w liczbach)",
    title_font=dict(size=20, color="#B2FF59", family="Arial"),  # Styl tytułu
    font=dict(color="#B2FF59"),  # Ogólny kolor czcionki
    xaxis=dict(
        tickfont=dict(size=14, color="#B2FF59"),  # Czcionka osi X
        title=dict(font=dict(size=16, color="#B2FF59"))  # Kolor tytułu osi X
    ),
    yaxis=dict(
        tickfont=dict(size=14, color="#B2FF59"),  # Czcionka osi Y
        title=dict(font=dict(size=16, color="#B2FF59"))  # Kolor tytułu osi Y
    ),
    legend=dict(
        title=dict(font=dict(size=14, color="#B2FF59")),  # Kolor tytułu legendy
        font=dict(size=12, color="#B2FF59")  # Kolor czcionki elementów legendy
    ),
    showlegend=True
)
bar_fig.update_traces(
    texttemplate='%{text}',  # Wyświetlanie liczby
    textposition='outside',
    textfont=dict(color="#B2FF59", size=14)  # Kolor i rozmiar tekstu nad słupkami
)
st.plotly_chart(bar_fig)

####################################    WYKRES 2 (KOŁOWY)
#

pie_fig = px.pie(
    same_cluster_df,
    names="Wiek",  # Kolumna używana do kategorii
    title="",  # Tytuł wykresu
    color_discrete_sequence=colors  # Wspólne kolory
)
pie_fig.update_layout(
    title=dict(
        text="Rozkład wieku w grupie (w procentach)",
        font=dict(size=20, color="#B2FF59", family="Arial")  # Rozmiar i kolor tytułu
    ),
    legend=dict(
        title=dict(text="Wiek", font=dict(size=14, color="#B2FF59")),  # Tytuł legendy
        font=dict(size=12, color="#B2FF59"),  # Elementy legendy
        x=1, y=1  # Pozycja legendy
    ),
    font=dict(color="#B2FF59"),  # Ogólny kolor czcionki
    showlegend=True
)
pie_fig.update_traces(
    textinfo='percent',  # Pokazuje procenty
    textposition='outside',
    textfont=dict(size=12, color="#B2FF59", weight="bold")  # Grubsza czcionka dla tekstu
)
st.plotly_chart(pie_fig)



####################################    WYKRES 3 
#
st.subheader("Rozkład wykształcenia w grupie (w liczbach i procentach)")

# Tworzenie dwóch kolumn w Streamlit
col1, col2 = st.columns(2)

# Kolumna 1: Histogram
with col1:
    fig = px.histogram(
        same_cluster_df, 
        x="Wykształcenie", 
        color="Wykształcenie",  # Dodanie kolorowania na podstawie wykształcenia
        color_discrete_sequence=colors  # Wspólne kolory
    )
    fig.update_layout(
        title="",
        title_font=dict(size=20, color="#B2FF59", family="Arial"),
        font=dict(color="#B2FF59"),
        xaxis_title="Wykształcenie",
        yaxis_title="Liczba osób",
        xaxis=dict(
            tickfont=dict(size=14, color="#B2FF59"),  # Czcionka osi X
            title=dict(font=dict(size=16, color="#B2FF59"))  # Kolor tytułu osi X
        ),
        yaxis=dict(
            tickfont=dict(size=14, color="#B2FF59"),  # Czcionka osi Y
            title=dict(font=dict(size=16, color="#B2FF59"))  # Kolor tytułu osi Y
        ),
        legend=dict(
            title=dict(font=dict(size=14, color="#B2FF59")),  # Kolor tytułu legendy
            font=dict(size=12, color="#B2FF59")  # Kolor czcionki elementów legendy
        ),
        showlegend=True
    )
    st.plotly_chart(fig)

# Kolumna 2: Wykres kołowy
with col2:
    # Grupowanie danych dla wykresu kołowego
    pie_data = same_cluster_df["Wykształcenie"].value_counts(normalize=True).reset_index()
    pie_data.columns = ["Wykształcenie", "Procent"]

    # Filtrowanie tylko wartości powyżej 0
    pie_data = pie_data[pie_data["Procent"] > 0]

    fig_pie = px.pie(
        pie_data, 
        values="Procent", 
        names="Wykształcenie", 
        color="Wykształcenie", 
        color_discrete_sequence=colors,  # Wspólne kolory
        title=" "
    )
    fig_pie.update_traces(
        textinfo="percent+value",  # Wyświetlanie procentów i wartości
        textposition="outside",  # Pozycja tekstu na zewnątrz
        texttemplate="%{percent:.2%}",  # Procenty z dwoma miejscami po przecinku
        textfont=dict(color="#B2FF59", size=14)  # Kolor i rozmiar tekstu
    )
    fig_pie.update_layout(
        title_font=dict(size=20, color="#B2FF59", family="Arial"),  # Styl tytułu
        font=dict(color="#B2FF59"),  # Ogólny kolor czcionki
        legend=dict(
            title=dict(font=dict(size=14, color="#B2FF59")),  # Styl tytułu legendy
            font=dict(size=12, color="#B2FF59")  # Styl czcionki elementów legendy
        ),
        showlegend=True  # Wyświetlanie legendy
    )
    st.plotly_chart(fig_pie)


####################################    Wykres 4
#
st.subheader("Rozkład ulubionych zwierząt w grupie (w liczbach i procentach)")


# Tworzenie dwóch kolumn w Streamlit
col1, col2 = st.columns(2)

# Kolumna 1: Histogram
with col1:
    fig = px.histogram(
        same_cluster_df, 
        x="Ulubione_zwierze", 
        color="Ulubione_zwierze",  # Dodanie kolorowania na podstawie danych
        color_discrete_sequence=colors  # Wspólne kolory
    )
    fig.update_layout(
        title="",
        title_font=dict(size=20, color="#B2FF59", family="Arial"),  # Styl tytułu
        font=dict(color="#B2FF59"),  # Ogólny kolor czcionki
        xaxis_title="Ulubione zwierzęta",
        yaxis_title="Liczba osób",
        xaxis=dict(
            tickfont=dict(size=14, color="#B2FF59"),  # Czcionka osi X
            title=dict(font=dict(size=16, color="#B2FF59"))  # Kolor tytułu osi X
        ),
        yaxis=dict(
            tickfont=dict(size=14, color="#B2FF59"),  # Czcionka osi Y
            title=dict(font=dict(size=16, color="#B2FF59"))  # Kolor tytułu osi Y
        ),
        legend=dict(
            title=dict(font=dict(size=14, color="#B2FF59")),  # Kolor tytułu legendy
            font=dict(size=12, color="#B2FF59")  # Kolor czcionki elementów legendy
        ),
        showlegend=True
    )
    st.plotly_chart(fig)

# Kolumna 2: Wykres kołowy
with col2:
    # Grupowanie danych dla wykresu kołowego
    pie_data = same_cluster_df["Ulubione_zwierze"].value_counts(normalize=True).reset_index()
    pie_data.columns = ["Ulubione_zwierze", "Procent"]

    # Filtrowanie tylko wartości powyżej 0
    pie_data = pie_data[pie_data["Procent"] > 0]

    fig_pie = px.pie(
        pie_data, 
        values="Procent", 
        names="Ulubione_zwierze", 
        color="Ulubione_zwierze", 
        color_discrete_sequence=colors,  # Wspólne kolory
        title=" "
    )
    fig_pie.update_traces(
        textinfo="percent+value",  # Wyświetlanie procentów i wartości
        textposition="outside",  # Pozycja tekstu na zewnątrz
        texttemplate="%{percent:.2%}",  # Procenty z dwoma miejscami po przecinku
        textfont=dict(color="#B2FF59", size=14)  # Kolor i rozmiar tekstu
    )
    fig_pie.update_layout(
        title_font=dict(size=20, color="#B2FF59", family="Arial"),  # Styl tytułu
        font=dict(color="#B2FF59"),  # Ogólny kolor czcionki
        legend=dict(
            title=dict(font=dict(size=14, color="#B2FF59")),  # Styl tytułu legendy
            font=dict(size=12, color="#B2FF59")  # Styl czcionki elementów legendy
        ),
        showlegend=False  # Wyświetlanie legendy
    )
    st.plotly_chart(fig_pie)


####################################    Wykres 5
#

st.subheader("Rozkład ulubionych miejsc w grupie (w liczbach i procentach)")

# Tworzenie dwóch kolumn dla wykresu "Ulubione_miejsce"
col1, col2 = st.columns(2)

# Kolumna 1: Histogram dla "Ulubione_miejsce"
with col1:
    fig = px.histogram(
        same_cluster_df, 
        x="Ulubione_miejsce", 
        color="Ulubione_miejsce",  # Dodanie kolorowania na podstawie danych
        color_discrete_sequence=colors  # Wspólne kolory
    )
    fig.update_layout(
        title="",
        title_font=dict(size=10, color="#B2FF59", family="Arial"),  # Styl tytułu
        font=dict(color="#B2FF59"),  # Ogólny kolor czcionki
        xaxis_title="Ulubione miejsce",
        yaxis_title="Liczba osób",
        xaxis=dict(
            tickfont=dict(size=14, color="#B2FF59"),  # Czcionka osi X
            title=dict(font=dict(size=16, color="#B2FF59"))  # Kolor tytułu osi X
        ),
        yaxis=dict(
            tickfont=dict(size=14, color="#B2FF59"),  # Czcionka osi Y
            title=dict(font=dict(size=16, color="#B2FF59"))  # Kolor tytułu osi Y
        ),
        legend=dict(
            title=dict(font=dict(size=14, color="#B2FF59")),  # Kolor tytułu legendy
            font=dict(size=12, color="#B2FF59")  # Kolor czcionki elementów legendy
        ),
        showlegend=True
    )
    st.plotly_chart(fig)

# Kolumna 2: Wykres kołowy dla "Ulubione_miejsce"
with col2:
    pie_data = same_cluster_df["Ulubione_miejsce"].value_counts(normalize=True).reset_index()
    pie_data.columns = ["Ulubione_miejsce", "Procent"]
    pie_data = pie_data[pie_data["Procent"] > 0]

    fig_pie = px.pie(
        pie_data, 
        values="Procent", 
        names="Ulubione_miejsce", 
        color="Ulubione_miejsce", 
        color_discrete_sequence=colors,  # Wspólne kolory
        title=" "
    )
    fig_pie.update_traces(
        textinfo="percent+value",  # Wyświetlanie procentów i wartości
        textposition="outside",  # Pozycja tekstu na zewnątrz
        texttemplate="%{percent:.2%}",  # Procenty z dwoma miejscami po przecinku
        textfont=dict(color="#B2FF59", size=14)  # Kolor i rozmiar tekstu
    )
    fig_pie.update_layout(
        title_font=dict(size=10, color="#B2FF59", family="Arial"),  # Styl tytułu
        font=dict(color="#B2FF59"),
        legend=dict(
            title=dict(font=dict(size=14, color="#B2FF59")),  # Styl tytułu legendy
            font=dict(size=12, color="#B2FF59")  # Styl czcionki elementów legendy
        ),
        showlegend=False
    )
    st.plotly_chart(fig_pie)

####################################    Wykres 6
#
st.subheader("Rozkład płci w grupie (w liczbach i procentach)")


# Tworzenie dwóch kolumn dla wykresu "Płeć"
col1, col2 = st.columns(2)

# Kolumna 1: Histogram dla "Płeć"
with col1:
    fig = px.histogram(
        same_cluster_df, 
        x="Płeć", 
        color="Płeć",  # Dodanie kolorowania na podstawie danych
        color_discrete_sequence=colors  # Wspólne kolory
    )
    fig.update_layout(
        title=" ",
        title_font=dict(size=20, color="#B2FF59", family="Arial"),  # Styl tytułu
        font=dict(color="#B2FF59"),  # Ogólny kolor czcionki
        xaxis_title="Płeć",
        yaxis_title="Liczba osób",
        xaxis=dict(
            tickfont=dict(size=14, color="#B2FF59"),  # Czcionka osi X
            title=dict(font=dict(size=16, color="#B2FF59"))  # Kolor tytułu osi X
        ),
        yaxis=dict(
            tickfont=dict(size=14, color="#B2FF59"),  # Czcionka osi Y
            title=dict(font=dict(size=16, color="#B2FF59"))  # Kolor tytułu osi Y
        ),
        legend=dict(
            title=dict(font=dict(size=14, color="#B2FF59")),  # Kolor tytułu legendy
            font=dict(size=12, color="#B2FF59")  # Kolor czcionki elementów legendy
        ),
        showlegend=True
    )
    st.plotly_chart(fig)

# Kolumna 2: Wykres kołowy dla "Płeć"
with col2:
    pie_data = same_cluster_df["Płeć"].value_counts(normalize=True).reset_index()
    pie_data.columns = ["Płeć", "Procent"]
    pie_data = pie_data[pie_data["Procent"] > 0]

    fig_pie = px.pie(
        pie_data, 
        values="Procent", 
        names="Płeć", 
        color="Płeć", 
        color_discrete_sequence=colors,  # Wspólne kolory
        title=" "
    )
    fig_pie.update_traces(
        textinfo="percent+value",  # Wyświetlanie procentów i wartości
        textposition="outside",  # Pozycja tekstu na zewnątrz
        texttemplate="%{percent:.2%}",  # Procenty z dwoma miejscami po przecinku
        textfont=dict(color="#B2FF59", size=14)  # Kolor i rozmiar tekstu
    )
    fig_pie.update_layout(
        title_font=dict(size=20, color="#B2FF59", family="Arial"),  # Styl tytułu
        font=dict(color="#B2FF59"),
        legend=dict(
            title=dict(font=dict(size=14, color="#B2FF59")),  # Styl tytułu legendy
            font=dict(size=12, color="#B2FF59")  # Styl czcionki elementów legendy
        ),
        showlegend=False
    )
    st.plotly_chart(fig_pie)