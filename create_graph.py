import matplotlib.pyplot as plt
import networkx as nx
from rdflib import Graph, Namespace, Literal, XSD
from rdflib.namespace import RDF, RDFS
import requests
from bs4 import BeautifulSoup
import re
import argparse


def fetch_page_content(url):
    response = requests.get(url)
    if response.status_code != 200:

        exit(1)
    return response.content


def parse_html_content(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    return soup.find('div', class_='mw-parser-output')


def extract_timeline_items(content):
    return content.find_all('h1')[0].find_all_next("ul")


def extract_events(timeline_items):
    events_list = []
    for item in timeline_items[9:]:
        for li in item.find_all('li', recursive=False):
            if li.a and li.a.text and ("BBY" in li.a.text or "ABY" in li.a.text):
                time = li.a.text

                if len(li.find_all('ul', recursive=False)) == 0:
                    continue
                events = li.find_all('ul', recursive=False)[
                    0].find_all('li', recursive=False)
                for event in events:
                    event_text = clean_event_text(event.text)
                    node_titles, nodes_hrefs = extract_node_info(event)
                    events_list.append(
                        [time, event_text, node_titles, nodes_hrefs])
    return events_list


def clean_event_text(text):
    text = re.sub(r'\s*\[.*?\]', '', text)
    return re.sub(r'\s+', ' ', text)


def extract_node_info(event):
    node_titles = []
    nodes_hrefs = []
    for a in event.find_all('a'):
        if not a.has_attr("title"):
            continue
        node_titles.append(a["title"])
        nodes_hrefs.append(a['href'])
    return node_titles, nodes_hrefs


def convert_star_wars_year(sw_year, reference_year=0):
    # Determine if it's BBY or ABY
    sw_year = sw_year.strip().replace(' ', '').replace(',', '')
    if 'BBY' in sw_year:
        return reference_year - int(sw_year.replace('BBY', ''))
    elif 'ABY' in sw_year:
        return reference_year + int(sw_year.replace('ABY', ''))
    return reference_year


def create_graph(events_list, filename='starwars_events_2.ttl', max_events=None):
    g = Graph()

    # Create a graph
    g = Graph()

    # Define namespaces
    SW = Namespace("https://starwars.fandom.com/")

    # Bind namespaces to prefixes
    g.bind("sw", SW)
    g.bind("rdf", RDF)
    g.bind("rdfs", RDFS)

    # Define the Event class and properties
    g.add((SW.Event, RDF.type, RDFS.Class))
    g.add((SW.Event, RDFS.label, Literal("Event")))
    g.add((SW.Event, RDFS.comment, Literal(
        "A significant occurrence in the Star Wars universe.")))

    g.add((SW.eventYear, RDF.type, RDF.Property))
    g.add((SW.eventYear, RDFS.domain, SW.Event))
    g.add((SW.eventYear, RDFS.range, XSD.integer))
    g.add((SW.eventYear, RDFS.label, Literal("Event Year")))
    g.add((SW.eventYear, RDFS.comment, Literal("The year of the event.")))

    g.add((SW.relatedTo, RDF.type, RDF.Property))  # New property name
    g.add((SW.relatedTo, RDFS.domain, SW.Event))
    g.add((SW.relatedTo, RDFS.range, SW.Thing))  # Link to the Thing class
    g.add((SW.relatedTo, RDFS.label, Literal("Related To")))
    g.add((SW.relatedTo, RDFS.comment, Literal(
        "Entities or concepts related to the event.")))

    g.add((SW.description, RDF.type, RDF.Property))
    g.add((SW.description, RDFS.domain, SW.Event))
    g.add((SW.description, RDFS.range, XSD.string))
    g.add((SW.description, RDFS.label, Literal("Description")))
    g.add((SW.description, RDFS.comment, Literal(
        "A brief description of the event.")))

    # Define the Thing class and properties
    g.add((SW.Thing, RDF.type, RDFS.Class))
    g.add((SW.Thing, RDFS.label, Literal("Thing")))
    g.add((SW.Thing, RDFS.comment, Literal(
        "An entity or concept related to an event.")))

    g.add((SW.name, RDF.type, RDF.Property))
    g.add((SW.name, RDFS.domain, SW.Thing))
    g.add((SW.name, RDFS.range, XSD.string))
    g.add((SW.name, RDFS.label, Literal("Name")))
    g.add((SW.name, RDFS.comment, Literal("The name of the thing.")))

    g.add((SW.url, RDF.type, RDF.Property))
    g.add((SW.url, RDFS.domain, SW.Thing))
    g.add((SW.url, RDFS.range, XSD.anyURI))
    g.add((SW.url, RDFS.label, Literal("URL")))
    g.add((SW.url, RDFS.comment, Literal("The URL of the thing.")))

    # Limit the number of events to max_events if specified
    if max_events is not None:
        events_list = events_list[:max_events]

    for i, event in enumerate(events_list):
        event_uri = SW["event_" + str(i)]
        g.add((event_uri, RDF.type, SW.Event))
        g.add((event_uri, RDFS.label, Literal(f"Event {i}")))
        g.add((event_uri, SW.eventYear, Literal(
            convert_star_wars_year(event[0]), datatype=XSD.integer)))
        g.add((event_uri, SW.description, Literal(
            event[1], datatype=XSD.string)))
        for title, href in zip(event[2], event[3]):
            thing_uri = SW[href[1:]]  # remove the leading /
            g.add((thing_uri, RDF.type, SW.Thing))
            g.add((thing_uri, RDFS.label, Literal(title)))
            g.add((thing_uri, SW.name, Literal(title, datatype=XSD.string)))
            # g.add((thing_uri, SW.url, Literal(href, datatype=XSD.anyURI)))
            g.add((event_uri, SW.relatedTo, thing_uri))
    # Serialize the graph to a file
    g.serialize(destination=filename, format='turtle')


if __name__ == "__main__":
    # argparse to get max_events and filename
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--max_events', type=int,
                        help='Maximum number of events to process')
    parser.add_argument('--filename', type=str,
                        help='Filename to save the graph')
    args = parser.parse_args()
    max_events = args.max_events if args.max_events else None
    filename = args.filename if args.filename else 'starwars_events.ttl'

    url = "https://starwars.fandom.com/wiki/Timeline_of_galactic_history"
    html_content = fetch_page_content(url)
    content = parse_html_content(html_content)
    timeline_items = extract_timeline_items(content)
    events_list = extract_events(timeline_items)

    create_graph(events_list, filename, max_events)
