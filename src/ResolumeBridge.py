import requests

RESOLUME_SERVER = "http://10.2.0.2:8080/api/v1"


def is_resolume_reachable():
    """Überprüft, ob der Resolume-Server erreichbar ist."""
    try:
        url = f"{RESOLUME_SERVER}/product"
        response = requests.get(url)
        if response.status_code == 200:
            print("Resolume-Server ist erreichbar.")
            return True
        else:
            print(f"Resolume-Server antwortet mit Status: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"Fehler beim Verbinden zum Resolume-Server: {e}")
        return False

def activate_layer(layer_id, column_id):
    """Aktiviert einen Clip auf einem Layer."""
    url = f"{RESOLUME_SERVER}/composition/layers/{layer_id}/clips/{column_id}/connect"
    try:
        response = requests.post(url)
        if response.status_code in (200, 204):
            print(f"Clip {column_id} auf Layer {layer_id} aktiviert")
        else:
            print(f"Fehler beim Aktivieren des Clips: {response.status_code}")
            return
    except requests.exceptions.RequestException as e:
        print(f"Fehler bei der Anfrage: {e}")


if __name__ == '__main__':
    # activate_layer(1, 4)
    # activate_layer(2, 4)
    # activate_layer(3, 4)
    is_resolume_reachable()
