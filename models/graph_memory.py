import os
import re
import time
from typing import List, Dict, Optional, Any

try:
    from falkordb import FalkorDB
    FALKORDB_AVAILABLE = True
except ImportError:
    FALKORDB_AVAILABLE = False


class GraphMemorySystem:
    """
    Sistema de Memoria Basado en Grafos de Conocimiento (Knowledge Graph / GraphRAG)
    usando FalkorDB (RedisGraph / openCypher).
    Diseñado para razonamiento relacional multi-hop de latencia ultra-baja y reducción de alucinaciones.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 6379, graph_name: str = "lily_knowledge_graph"):
        self.host = host
        self.port = port
        self.graph_name = graph_name
        self.client: Optional[Any] = None
        self.graph: Optional[Any] = None
        self.is_connected = False

        self._init_connection()

    def _init_connection(self):
        """Intenta conectar a FalkorDB con tiempo de espera controlado."""
        if not FALKORDB_AVAILABLE:
            print("[FalkorDB] Módulo 'falkordb' no disponible en el entorno Python.")
            return

        try:
            # FalkorDB client
            self.client = FalkorDB(host=self.host, port=self.port, socket_timeout=1.5, socket_connect_timeout=1.5)
            # Prueba de conectividad rápida
            self.graph = self.client.select_graph(self.graph_name)
            self.graph.query("RETURN 1")
            self.is_connected = True
            self._init_indices()
            print(f"[FalkorDB] Conectado exitosamente al grafo '{self.graph_name}' en {self.host}:{self.port}")
        except Exception as e:
            self.is_connected = False
            self.graph = None
            # Silencioso o informativo para no bloquear a Lily si Docker/FalkorDB no está encendido
            print(f"[FalkorDB] Servicio no detectado en {self.host}:{self.port} ({e}). Modo fallback activo (SQLite + ChromaDB).")

    def _init_indices(self):
        """Crea índices para búsquedas rápidas en nodos principales."""
        if not self.is_connected or not self.graph:
            return
        try:
            self.graph.query("CREATE INDEX FOR (u:User) ON (u.name)")
            self.graph.query("CREATE INDEX FOR (e:Entity) ON (e.name)")
        except Exception:
            pass

    def check_connection(self) -> bool:
        """Verifica si la conexión sigue viva o intenta reconectar si estaba inactiva."""
        if self.is_connected and self.graph:
            try:
                self.graph.query("RETURN 1")
                return True
            except Exception:
                self.is_connected = False

        # Intentar reconectar
        try:
            self._init_connection()
            return self.is_connected
        except Exception:
            return False

    def add_triple(
        self,
        subject: str,
        predicate: str,
        object_: str,
        subject_type: str = "User",
        object_type: str = "Entity",
        properties: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Inserta o actualiza una tripleta (Sujeto)-[PREDICADO]->(Objeto) en el grafo de conocimiento.
        """
        if not self.check_connection():
            return False

        # Limpiar nombre del predicado para compatibilidad Cypher
        pred_clean = re.sub(r'[^A-Za-z0-9_]', '_', predicate.upper().strip())
        subj_type_clean = re.sub(r'[^A-Za-z0-9_]', '_', subject_type)
        obj_type_clean = re.sub(r'[^A-Za-z0-9_]', '_', object_type)

        query = f"""
        MERGE (s:{subj_type_clean} {{name: $s_name}})
        MERGE (o:{obj_type_clean} {{name: $o_name}})
        MERGE (s)-[r:{pred_clean}]->(o)
        SET r.updated_at = $timestamp
        RETURN type(r)
        """
        params = {
            "s_name": subject.strip(),
            "o_name": object_.strip(),
            "timestamp": int(time.time())
        }

        try:
            self.graph.query(query, params)
            return True
        except Exception as e:
            print(f"[FalkorDB] Error al guardar tripleta ({subject})-[:{predicate}]->({object_}): {e}")
            return False

    def query_context_for_message(self, user_message: str, user_name: str = "Mijin") -> List[str]:
        """
        Busca hechos relevantes en el grafo de conocimiento relacionados con el mensaje del usuario.
        Retorna una lista de cadenas legibles que representan hechos conocidos en el grafo.
        """
        if not self.check_connection():
            return []

        facts: List[str] = []
        clean_text = user_message.lower()

        # 1. Obtener hechos directos del usuario
        try:
            query_user_facts = """
            MATCH (u:User {name: $user_name})-[r]->(o)
            RETURN type(r) AS rel, o.name AS target, labels(o) AS target_type
            LIMIT 15
            """
            result = self.graph.query(query_user_facts, {"user_name": user_name})

            for row in result.result_set:
                rel = row[0]
                target = row[1]
                target_str = str(target)

                # Si el target aparece en el mensaje o si es una consulta sobre preferencias/vida
                is_relevant = any(w in clean_text for w in [target_str.lower(), rel.lower(), "cómo", "como", "quién", "quien", "cuál", "cual", "recuerdas", "sabes", "tengo", "mi", "mis"])
                fact_str = f"{user_name} -> {rel.replace('_', ' ')} -> {target}"

                if is_relevant:
                    facts.append(fact_str)
        except Exception as e:
            print(f"[FalkorDB] Error al consultar hechos del usuario: {e}")

        # 2. Búsqueda semántica/léxica de entidades mencionadas en el mensaje y sus conexiones (Multi-hop)
        try:
            words = [w for w in re.findall(r'\b[a-zA-ZáéíóúÁÉÍÓÚñÑ]{3,}\b', clean_text)]
            if words:
                query_entities = """
                MATCH (e:Entity)-[r]-(other)
                WHERE toLower(e.name) IN $keywords
                RETURN e.name AS e_name, type(r) AS rel, other.name AS other_name
                LIMIT 10
                """
                res_ent = self.graph.query(query_entities, {"keywords": words})
                for row in res_ent.result_set:
                    f = f"{row[0]} <-> {str(row[1]).replace('_', ' ')} <-> {row[2]}"
                    if f not in facts:
                        facts.append(f)
        except Exception as e:
            print(f"[FalkorDB] Error al consultar entidades relacionadas: {e}")

        return list(dict.fromkeys(facts))[:8]  # Deduplicar preservando orden, máximo 8 hechos

    def extract_and_store_facts(self, message: str, user_name: str = "Mijin") -> int:
        """
        Extrae automáticamente hechos y relaciones del mensaje del usuario y los guarda en el grafo.
        Retorna la cantidad de tripletas guardadas.
        """
        if not self.check_connection():
            return 0

        stored = 0
        text = message.strip()

        # Patrones de extracción deterministas basados en lenguaje natural
        patterns = [
            # Mascotas: "mi perro se llama Toby", "tengo un gato llamado Michi", "mi mascota es un loro llamado Pepe"
            (r'\bmi\s+(perro|gata|gato|mascota|loro|conejo|hamster)\s+(?:se llama|es)\s+([A-Za-z0-9áéíóúñÁÉÍÓÚÑ]+)',
             lambda m: (user_name, f"TIENE_{m.group(1).upper()}", m.group(2).capitalize(), "User", "Mascota")),

            (r'\btengo\s+(?:un|una)\s+(perro|gata|gato|mascota)\s+(?:llamado|llamada)\s+([A-Za-z0-9áéíóúñÁÉÍÓÚÑ]+)',
             lambda m: (user_name, f"TIENE_{m.group(1).upper()}", m.group(2).capitalize(), "User", "Mascota")),

            # Familiares y amigos: "mi amigo se llama Carlos", "mi hermano se llama Juan"
            (r'\bmi\s+(amigo|amiga|hermano|hermana|papá|mama|mamá|padre|madre|novia|novio|esposa|esposo|primo|prima)\s+(?:se llama|es)\s+([A-Za-z0-9áéíóúñÁÉÍÓÚÑ]+)',
             lambda m: (user_name, f"TIENE_{m.group(1).upper()}", m.group(2).capitalize(), "User", "Persona")),

            # Gustos y preferencias: "me gusta el rock", "me encanta la pizza", "me apasiona programar"
            (r'\b(?:me encanta|me gusta mucho|me gusta|me apasiona)\s+(?:el|la|los|las)?\s*([A-Za-z0-9áéíóúñÁÉÍÓÚÑ\s]{3,30}?)(?=[,.\n]|$)',
             lambda m: (user_name, "LE_GUSTA", m.group(1).strip().capitalize(), "User", "Gusto")),

            # Desagrados: "odio el café", "no me gusta el reggaeton", "detesto madrugar"
            (r'\b(?:odio|detesto|no me gusta|no me agrada)\s+(?:el|la|los|las)?\s*([A-Za-z0-9áéíóúñÁÉÍÓÚÑ\s]{3,30}?)(?=[,.\n]|$)',
             lambda m: (user_name, "NO_LE_GUSTA", m.group(1).strip().capitalize(), "User", "Disgusto")),

            # Residencia / Ubicación: "vivo en Madrid", "me mudé a Tokio", "soy de México"
            (r'\b(?:vivo en|me mudé a|estoy viviendo en)\s+([A-Za-z0-9áéíóúñÁÉÍÓÚÑ\s]{3,25}?)(?=[,.\n]|$)',
             lambda m: (user_name, "VIVE_EN", m.group(1).strip().capitalize(), "User", "Lugar")),

            # Profesión / Ocupación: "trabajo como programador", "trabajo de diseñador", "soy ingeniero"
            (r'\b(?:trabajo como|trabajo de|mi trabajo es|soy)\s+(?:un|una)?\s*([A-Za-z0-9áéíóúñÁÉÍÓÚÑ\s]{4,30}?)(?=[,.\n]|$)',
             lambda m: (user_name, "TRABAJA_COMO", m.group(1).strip().capitalize(), "User", "Profesion")),

            # Cumpleaños: "mi cumpleaños es el 15 de marzo", "cumplo años el 2 de mayo"
            (r'\b(?:mi cumpleaños es el|cumplo años el)\s+([0-9]{1,2}\s+de\s+[A-Za-z]+)',
             lambda m: (user_name, "CUMPLEAÑOS", m.group(1).strip(), "User", "Fecha"))
        ]

        for pattern, extractor in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    s, p, o, st, ot = extractor(match)
                    if s and p and o and len(o) > 1:
                        if self.add_triple(s, p, o, subject_type=st, object_type=ot):
                            print(f"[FalkorDB GraphRAG] Hecho aprendido en grafo: ({s})-[:{p}]->({o})")
                            stored += 1
                except Exception as e:
                    print(f"[FalkorDB] Error procesando patrón de extracción: {e}")

        return stored

    def get_graph_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas sobre el número de nodos y aristas en el grafo."""
        if not self.check_connection():
            return {"status": "offline", "nodes": 0, "relationships": 0}

        try:
            nodes_res = self.graph.query("MATCH (n) RETURN count(n) AS cnt")
            rel_res = self.graph.query("MATCH ()-[r]->() RETURN count(r) AS cnt")
            nodes_cnt = nodes_res.result_set[0][0] if nodes_res.result_set else 0
            rel_cnt = rel_res.result_set[0][0] if rel_res.result_set else 0
            return {
                "status": "online",
                "graph_name": self.graph_name,
                "nodes": nodes_cnt,
                "relationships": rel_cnt
            }
        except Exception as e:
            return {"status": "error", "error": str(e), "nodes": 0, "relationships": 0}
