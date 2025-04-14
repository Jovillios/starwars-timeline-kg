# SW Timeline KG

## Queries

Get all things in the Star Wars universe with their names.

```sparql
prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#>
prefix sw: <https://starwars.fandom.com>
prefix xsd: <http://www.w3.org/2001/XMLSchema#>

SELECT ?thing ?name WHERE {
    ?thing rdf:type sw:Thing ;
           sw:name ?name .
}
```

Get all events in the Star Wars universe with their descriptions and dates.

```sparql
prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#>
prefix sw: <https://starwars.fandom.com>
prefix xsd: <http://www.w3.org/2001/XMLSchema#>

SELECT ?event ?description ?eventYear WHERE {
    ?event rdf:type sw:Event ;
    sw:description ?description ;
    sw:eventYear ?eventYear .
    FILTER (xsd:integer(?eventYear) >= 0)
}
```
