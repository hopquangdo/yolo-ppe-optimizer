# Running the Edge Agent

You run one **edge-agent** container per edge device. It connects to the central
MQTT broker, sends heartbeats, receives commands (deploy model, update config,
restart, diagnostics), runs the camera → PPE pipeline, and reports violations.
A small HTTP API on port **9100** lets you check it locally.

You need three things from whoever operates the platform:

1. the **image reference** — e.g. `registry.example.com/ai-edge-ops/edge-agent:1.0`
   (for a Jetson, the `:1.0-jetson` tag);
2. the **MQTT broker** address (host, port, and credentials / TLS files if any);
3. the **server API** base URL — e.g. `https://ops.example.com/api/v1`.

Nothing else — no source, no build.

---

## 1. Register the device (platform operator, once)

On the server:

```bash
curl -X POST https://ops.example.com/api/v1/fleet/devices \
  -H 'content-type: application/json' \
  -d '{"name":"cam-plant-a-01","site":"Plant A","hardware":"Jetson Orin Nano"}'
# -> { "success": true, "data": { "id": 42, ... } }
```

The returned **`id`** is what goes in `DEVICE_ID` below.

---

## 2. On the device — two files

### `docker-compose.yml`

```yaml
name: ai-edge-ops-edge

services:
  edge-agent:
    image: registry.example.com/ai-edge-ops/edge-agent:1.0
    env_file: .env
    ports:
      - "9100:9100"
    volumes:
      - edge_state:/var/lib/edge-agent            # desired/reported state + downloaded models
      # - ./certs:/certs:ro                        # if using MQTT TLS
    restart: unless-stopped
    # --- Jetson: GPU + camera passthrough (use the :1.0-jetson image) ---
    # runtime: nvidia
    # devices:
    #   - "/dev/video0:/dev/video0"

volumes:
  edge_state:
```

### `.env`

```dotenv
# identity — the numeric id from step 1
DEVICE_ID=42
TRANSPORT=mqtt
LOG_LEVEL=INFO

# central MQTT broker
MQTT__HOST=mqtt.example.com
MQTT__PORT=1883
MQTT__USERNAME=edge
MQTT__PASSWORD=change-me
MQTT__TLS=false
# for TLS (port 8883):
# MQTT__TLS=true
# MQTT__TLS_CA_FILE=/certs/ca.crt
# MQTT__TLS_CERT_FILE=/certs/device.crt
# MQTT__TLS_KEY_FILE=/certs/device.key

# server API (enrollment / HTTP fallback)
SERVER__BASE_URL=https://ops.example.com/api/v1

# one camera (add CAMERAS__1__*, CAMERAS__2__* for more)
CAMERAS__0__ID=bay-1
CAMERAS__0__SOURCE=0                 # webcam index | rtsp://user:pass@host/stream | /media/clip.mp4
CAMERAS__0__ZONE=loading-bay
CAMERAS__0__FPS_LIMIT=10

# inference tuning (optional)
INFERENCE__CONF_THRESHOLD=0.25
INFERENCE__PPE_REQUIRED=["helmet","vest"]
INFERENCE__IMGSZ=640
```

> If `CAMERAS__0__SOURCE` is a file path or `/dev/video*`, mount it into the
> container (`volumes:` / `devices:`).

---

## 3. Start

```bash
docker compose pull        # get the image
docker compose up -d
```

Check it:

```bash
curl http://localhost:9100/status
```

```json
{
  "success": true,
  "data": {
    "device_id": "42",
    "runtime_status": "idle",
    "active_model_version_id": null,
    "desired_model_version_id": null,
    "in_sync": true,
    "runtime_loaded": false,
    "cameras": ["bay-1"]
  }
}
```

On the server, the device should now show `online`:

```bash
curl https://ops.example.com/api/v1/fleet/devices
```

---

## 4. Deploying a model (platform operator)

The device pulls it automatically once you activate a deployment:

```bash
# after a published model version exists with an artifact_uri:
curl -X POST https://ops.example.com/api/v1/deployments \
  -H 'content-type: application/json' \
  -d '{"device_id":42,"model_version_id":7,"activate_now":true}'
```

The server sends a `deploy_model` command; the agent downloads the artifact
(checksum-verified), loads it, and reports back. Follow along:

```bash
curl http://localhost:9100/status     # runtime_status: applying -> active
```

---

## 5. Local API — `http://<device>:9100`

| Method | Path | |
|---|---|---|
| GET | `/health` | liveness probe |
| GET | `/status` | runtime state, active vs desired model, `in_sync`, cameras |
| GET | `/config` | current inference config |
| PUT | `/config` | local override (until the server pushes a new config) |
| GET | `/cameras` | configured cameras |
| POST | `/model/reconcile` | re-apply the desired model now |

---

## 6. Operating

```bash
docker compose logs -f edge-agent
docker compose restart edge-agent
docker compose pull && docker compose up -d      # upgrade to a new image tag
docker compose down                               # stop, keep state
docker compose down -v                            # stop, wipe local state + models
```

The agent reconnects automatically. If the device drops, the broker's last-will
marks it `offline` on the server within the platform's offline threshold
(default 90 s).

---

## 7. Troubleshooting

| Symptom | Check |
|---|---|
| device stays `offline` on server | `MQTT__HOST/PORT` reachable from the container; `DEVICE_ID` is the **numeric** id; `docker compose logs` for connection errors |
| `runtime_status: failed` after deploy | artifact URL reachable from the device; checksum; disk space in the `edge_state` volume |
| no violations | camera opening? (`logs`); `CAMERAS__0__SOURCE` correct; a model is `active` (`/status`) |
| TLS handshake errors | cert paths mounted and readable; `MQTT__PORT=8883`; CA matches the broker |
