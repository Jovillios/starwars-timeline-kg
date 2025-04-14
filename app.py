import streamlit as st
from rdflib import Graph
import pandas as pd

# Title of the app
st.title("RDF Query App")

# Text area for SPARQL query input
query = st.text_area("Enter your SPARQL query:", height=150)

# File uploader for RDF data
rdf_file = st.file_uploader("Upload an RDF file (e.g., .ttl, .rdf, .xml)", type=[
                            "ttl", "rdf", "xml", "n3"])

if rdf_file and query.strip() != "":
    # Load the RDF data into a graph
    g = Graph()
    g.parse(file=rdf_file)

    # Execute the query
    results = g.query(query)

    # Convert results to a pandas DataFrame
    data = []
    for row in results:
        data.append([str(val) for val in row])

    df = pd.DataFrame(data, columns=[str(var) for var in results.vars])

    # Display the results
    st.subheader("Query Results")
    st.dataframe(df)

else:
    st.info("Please upload an RDF file and enter a SPARQL query.")
