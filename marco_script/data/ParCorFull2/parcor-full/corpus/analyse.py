#############################################
###### Lien Git-Hub #########################
# https://github.com/chardmeier/parcor-full #
#############################################
#############################################

DEBUG_MOT = True
DEBUG_COREF = False

CHEMIN_DATA = "./parcor-full/corpus/TED/FR/"
MARKABLE = "Markables/"
NOM_FICHIER = "000_779"

BASEDATA = "{0}{1}_words.xml".format("Basedata/", NOM_FICHIER)
SOURCEXML = "{0}{1}".format(CHEMIN_DATA, BASEDATA)
CHAINECOREFXML = "{0}{1}{2}_coref_level.xml".format(CHEMIN_DATA, MARKABLE, NOM_FICHIER)


class Mot:
    def __init__(self, id, token) -> None:
        """Générateur d'un mot

        Args:
            id (int): ID du mot
            token (String): token représentant le mot
        """
        self._id = id
        self._token = token

    def __repr__(self) -> str:
        """Retourne un affiche du mot au format TSV

        Returns:
            str: Représentation du mot au format TSV
        """
        return "{0}\t{1}".format(self._id, self._token)

    def __str__(self) -> str:
        """Retourne une chaine de caractère représentant le mot au format TSV

        Returns:
            str: Représentation du mot au format TSV
        """
        return "{0}\t{1}".format(self._id, self._token)


def recup_mot(Nom_Fichier):
    """Récupère la liste de mot du document

    Args:
        Nom_Fichier (String): chaine de caractères représentant le chemin et le fichier à étudier

    Returns:
        List: Liste de Mot
    """
    import xml.etree.cElementTree as ET

    liste_mot = []
    for word in ET.parse(Nom_Fichier).getroot():  # Parcours les éléments
        liste_mot.append(Mot(word.attrib["id"], word.text))  # Puis créé une liste de Mot
    return liste_mot


if DEBUG_MOT:
    liste_mot = recup_mot(SOURCEXML)
    print("\n".join(map(str, liste_mot)))


class Mention:
    def __init__(self, id, span) -> None:
        self._id = id
        self._span = span


def recup_coref(Nom_Fichier):
    """Récupère les coréférences présentes dans le document

    Args:
        Nom_Fichier (String): chemin + nom + type du fichier où lire les données
    """
    import xml.etree.cElementTree as ET

    liste_mot = []
    i = 1
    for word in ET.parse(Nom_Fichier).getroot():  # Parcours les éléments
        print("**************")
        print(i)
        print(word.attrib["id"])
        print(word.attrib["span"])
        print(word.attrib["coref_class"])
        print(word.attrib["mmax_level"])
        # print(word.attrib["nptype"])
        # print(word.attrib["antetype"])
        print(word.attrib["mention"])
        i += 1
    return liste_mot


if DEBUG_COREF:
    print(recup_coref(CHAINECOREFXML))
