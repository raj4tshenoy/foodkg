from rdflib import Graph, URIRef, Literal
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_foodon_graph():
    # Load the FoodOn ontology
    g = Graph()
    g.parse("https://raw.githubusercontent.com/FoodOntology/foodon/master/foodon.owl", format="xml")
    return g

def get_foodon_triples(graph):
    triples = []
    for s, p, o in graph:
        if isinstance(s, URIRef) and isinstance(p, URIRef):
            # Filter out literals with datetime or other unsupported formats
            if isinstance(o, Literal):
                try:
                    if o.datatype and 'dateTime' in o.datatype:
                        continue
                    if isinstance(o.value, (str, int, float)):
                        triples.append((str(s), str(p), str(o)))
                except Exception as e:
                    logger.warning(f"Skipping literal with parsing issue: {o} - {e}")
            else:
                triples.append((str(s), str(p), str(o)))
    return triples

def map_uris_to_indices(triples):
    entity2id = {}
    relation2id = {}
    entity_idx = 0
    relation_idx = 0

    indexed_triples = []

    for s, p, o in triples:
        if s not in entity2id:
            entity2id[s] = entity_idx
            entity_idx += 1
        if o not in entity2id:
            entity2id[o] = entity_idx
            entity_idx += 1
        if p not in relation2id:
            relation2id[p] = relation_idx
            relation_idx += 1
        
        indexed_triples.append((entity2id[s], relation2id[p], entity2id[o]))

    return indexed_triples, entity2id, relation2id
