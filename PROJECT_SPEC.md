# PROJECT_SPEC.md
## Sirius — App de chat por texto y voz (estilo Gemini) para uso infantil

> Este documento es el contrato de construcción para el agente autónomo (OpenHands + Ornith).
> Colócalo en la raíz del repo de Bitbucket (`HumberDJ/sirius`). Aliméntalo al agente
> **hito por hito**, no de una sola vez. Este repo es independiente del repo de entrenamiento
> del modelo (`F:\LLM\Proyectos\Stardusts`) — este documento ya extrae de ahí todo lo que
> el agente necesita saber, así que no hace falta darle también `PROJECT_CONTEXT.md`.

---

## 1. Resumen del proyecto

Construir **Sirius**, una web app local (React + backend de voz) donde las usuarias (niñas,
6-12 años) puedan hablar o escribir con un LLM local afinado (SFT+DPO) desde su celular,
dentro de la red doméstica, con una interfaz cuyo elemento central es un **orbe orgánico con
gradiente de color que reacciona en tiempo real al audio** — el mismo lenguaje visual que
Gemini popularizó para "hablar con una IA", construido con técnicas propias (sección 7),
no assets ni gradientes copiados de ningún producto comercial.

El proyecto tiene dos caras igual de importantes: la experiencia de las niñas (educativa,
respetuosa, de apoyo didáctico y entretenimiento seguro) y un panel de administración para
el padre, donde se puede revisar cualquier detalle de las conversaciones que tuvieron.

**Usuarias objetivo:** niñas de 6-12 años. La UI debe ser simple, sin texto denso, con
retroalimentación visual clara en cada estado.

**Administrador:** el padre, con acceso separado a un panel de supervisión.

---

## 2. Contexto técnico heredado (del proyecto de entrenamiento del modelo)

Esta sección resume lo que ya existe y está terminado, para que el agente no lo cuestione,
lo reconstruya, ni asuma parámetros distintos.

| Dato | Valor |
|---|---|
| Modelo base | Qwen2-7B-Instruct |
| Fine-tuning | SFT (QLoRA 4-bit) → DPO (TRL, beta=0.1), ambas fases completadas y verificadas |
| Exportación | GGUF Q4_K_M, ~4.9 GB |
| Idiomas | Español (principal), inglés (secundario/educativo) — el modelo ya es bilingüe |
| Categorías de seguridad ya alineadas vía DPO | violencia/autodaño, contenido sexual, actividades ilegales, bullying, desinformación peligrosa, privacidad con extraños, manipulación emocional, sustancias, armas/explosivos, evasión de reglas (10 categorías, 200 pares) |
| Comportamiento verificado | El modelo ya niega amablemente preguntas peligrosas y redirige a temas seguros — esto es la **primera capa** de seguridad, no la única (ver sección 9.6) |
| Hardware donde correrá en producción | AMD Ryzen 9 9900X3D, 64GB DDR5, RTX 5080 16GB VRAM, Windows 11 + WSL2 |
| Implicación de hardware | El modelo (4.9GB) cabe completo en los 16GB de VRAM sin offload a CPU — a diferencia de Ornith (35B), que sí lo necesita. Esto se traduce en una latencia esperada baja (ver 14) |

---

## 3. Alcance y no-alcance

**Dentro de alcance:**
- Chat por texto y por voz con el modelo local vía llama-server.
- Interfaz visual con el orbe reactivo (sección 7) y sistema de diseño fijo.
- Perfiles individuales por niña (sección 10).
- Panel de administración parental con revisión completa de conversaciones (sección 11).
- Módulos educativos y de aprendizaje (sección 12).
- Accesibilidad (sección 13).
- Acceso desde celular dentro de la red local (vía Tailscale).
- Integración obligatoria con el pipeline de seguridad (detector de crisis + juez de seguridad,
  sección 9) — **adicional** a lo que el modelo ya trae alineado por DPO, no un sustituto.

**Fuera de alcance (el agente NO debe hacer esto sin preguntar):**
- Ninguna llamada a APIs externas (OpenAI, Google, ElevenLabs, etc.). Todo el procesamiento es local.
- Ninguna exposición a internet público (nada de puertos abiertos al WAN, nada de ngrok/cloudflared).
- No introducir autenticación de terceros, analytics, telemetría, ni CDNs externos más allá de los
  estrictamente necesarios para dependencias de build.
- No cambiar el modelo, el motor de inferencia (llama-server), ni reentrenar nada — eso ya
  está resuelto en el otro repositorio.
- No decidir arquitectura de seguridad por su cuenta: si algo no está cubierto en la sección 9, debe
  detenerse y dejar un TODO explícito en vez de improvisar.
- No implementar la IA como un "amigo" o personaje que finge ser humano, ni roleplay de relación
  personal (ver sección 9.4). Este es un principio de diseño, no un detalle de copy.
- No usar gamificación de tipo "diseño adictivo" (rachas, notificaciones de retorno, urgencia
  artificial) — ver sección 12.5.
- No copiar assets, gradientes exactos, tipografía ni wordmarks de Gemini o cualquier otro
  producto comercial. Se replica la **técnica y el tipo de interacción**, con identidad visual
  propia (sección 7.1).
- No inventar estructura de carpetas, contratos de datos o esquema de base de datos fuera de
  la sección 6 sin dejarlo marcado como decisión pendiente de tu revisión.

---

## 4. Stack tecnológico y configuración

### 4.1 Tabla de stack

| Capa | Tecnología | Propósito |
|---|---|---|
| Frontend | React + Vite + TypeScript | SPA, mobile-first |
| Animación de UI / transiciones | Framer Motion | transiciones entre estados, layout, chrome de la app (no el render del orbe en sí) |
| Máquina de estados | XState v5 + `@xstate/react` | FSM explícita de los 7 estados (sección 8) |
| Render del orbe | Canvas 2D nativo + `simplex-noise` | técnica orgánica de la sección 7.2 — no hay librería "todo en uno" para esto |
| Interpolación de color | `chroma-js` | transiciones de gradiente suaves entre estados |
| Audio en tiempo real | Web Audio API (`AnalyserNode`) — nativo del navegador | reactividad del orbe al volumen real (sección 7.3) |
| Comunicación en vivo | WebSocket | contrato definido en sección 6.2 |
| Orquestación de voz (backend) | Pipecat | pipeline STT → LLM → TTS, sección 8 del pipeline en 6.2 |
| STT | faster-whisper | `<!-- TODO: elegir tamaño (base/small) tras medir latencia real -->` |
| VAD | Silero VAD | detección de inicio/fin de habla |
| TTS | Kokoro (preferido) o Piper (fallback) | debe soportar español; si hay voces bilingües disponibles, mejor, dado que el modelo también responde en inglés |
| LLM (producción) | Qwen2-7B-Instruct + SFT + DPO, GGUF Q4_K_M, vía llama-server OpenAI-compatible | **no es Ornith** — ver 4.2 |
| Persistencia | SQLite local | esquema completo en sección 6.3 |
| Seguridad | Detector de crisis pre-LLM + juez de seguridad (Gemma4) | ver sección 9 |

No se permite sustituir ninguna pieza de esta tabla por "una alternativa más fácil de instalar"
sin dejarlo documentado como decisión explícita en el README, citando el motivo.

### 4.2 Nota importante — Ornith no es el modelo de producción

Ornith es el agente que construye esta app (a través de OpenHands); el modelo que la app usa
cuando las niñas conversan con ella es Qwen2-7B-Instruct + SFT + DPO (sección 2). Ambos
comparten el mismo motor (llama-server) pero no cargan al mismo tiempo en este hardware, así
que el endpoint del LLM de producción se resuelve por configuración (4.3), nunca hardcodeado.

### 4.3 Configuración del endpoint del LLM en tiempo de ejecución

El backend (Pipecat) lee la conexión al LLM desde variables de entorno, nunca escritas
directamente en el código:

```bash
# .env (no se sube a git — .env.example sí, como plantilla)
LLM_BASE_URL=http://localhost:8081/v1
LLM_MODEL_NAME=qwen2-7b-kidsafe
LLM_API_KEY=none
```

---

## 5. Arquitectura general

```
[Celular de la niña] --HTTPS/WSS (Tailscale)--> [Frontend Vite/React]
                                                        |
                                                        | WebSocket (sección 6.2)
                                                        v
                                          [Backend Pipecat]
                                          Silero VAD -> faster-whisper (STT)
                                                |
                                                v
                                    [Detector de crisis pre-LLM] --(bloqueo)--> respuesta segura
                                                |                                    |
                                                v                                    |
                                    [Qwen2-7B kidsafe vía llama-server]              |
                                                |                                    |
                                                v                                    |
                                    [Juez de seguridad] --(bloqueo)--> respuesta segura
                                                |                                    |
                                                v                                    v
                                          Kokoro/Piper (TTS)          [SQLite local: sección 6.3]
                                                |                                    ^
                                [Audio + evento "hablando"] -> Frontend              |
                                                                                      |
                                                          [Panel parental] <----------+
                                                          (autenticación propia, sección 11.1)
```

El chat de **texto** usa el mismo camino desde el detector de crisis en adelante, saltándose
solo STT/TTS. Cada turno se escribe en SQLite después de pasar por el juez de seguridad.

---

## 6. Arquitectura de componentes y contratos de datos

Esto se fija ahora para que cada hito construya sobre la misma base, no una inventada
sobre la marcha.

### 6.1 Estructura de carpetas del frontend

```
src/
├── components/
│   ├── orb/
│   │   ├── VoiceOrb.tsx          # Canvas 2D, técnica de la sección 7.2
│   │   ├── useOrbAnimation.ts    # hook: ruido simplex + loop de rAF
│   │   └── orbPalettes.ts        # colores/gradientes por estado (sección 7.1, 7.4)
│   ├── chat/
│   │   ├── ChatInput.tsx
│   │   ├── MessageBubble.tsx     # complemento textual a la voz
│   │   └── Captions.tsx          # subtítulos sincronizados (sección 13)
│   ├── profile/
│   │   ├── ProfileSelector.tsx
│   │   └── ProfileAvatar.tsx
│   ├── parental/
│   │   ├── ParentalGate.tsx      # PIN/contraseña, sección 11.1
│   │   ├── ConversationLog.tsx   # sección 11.2
│   │   ├── SafetyEventsLog.tsx   # sección 11.3
│   │   └── ProfileSettings.tsx   # sección 11.4
│   └── shared/
│       ├── ErrorBanner.tsx
│       └── WindingDownBanner.tsx
├── hooks/
│   ├── useAudioAnalyser.ts       # wrapper de AnalyserNode (sección 7.3)
│   ├── useWebSocket.ts
├── machines/
│   └── conversationMachine.ts    # definición XState, sección 8
├── lib/
│   └── noise.ts                  # wrapper de simplex-noise
├── styles/
│   └── tokens.css                # variables de la sección 7.1/7.5
└── App.tsx
```

### 6.2 Contrato WebSocket (cliente ↔ servidor)

Definir estos tipos primero (ej. en un archivo `shared-types.ts` que importen ambos lados
si el backend también es TypeScript, o como documentación equivalente en Python) y no
desviarse de esta forma sin actualizar este documento:

```typescript
// Cliente → Servidor
type ClientMessage =
  | { type: "user_text"; profileId: string; text: string }
  | { type: "audio_chunk"; profileId: string; data: ArrayBuffer } // PCM 16kHz mono
  | { type: "audio_end"; profileId: string };

// Servidor → Cliente
type ServerMessage =
  | { type: "state_change"; state: OrbState }               // sección 8
  | { type: "assistant_text"; text: string; blocked: boolean }
  | { type: "audio_chunk"; data: ArrayBuffer }               // TTS, PCM
  | { type: "transcript_partial"; text: string }             // subtítulos en vivo, sección 13
  | { type: "error"; message: string };
```

### 6.3 Esquema de base de datos (SQLite)

```sql
CREATE TABLE profiles (
  id TEXT PRIMARY KEY,
  nombre TEXT NOT NULL,
  edad INTEGER NOT NULL,
  avatar_id TEXT NOT NULL,
  color_acento TEXT NOT NULL,
  intereses TEXT,                          -- JSON array, sección 10
  limite_diario_min INTEGER NOT NULL DEFAULT 30,
  horario_nocturno_inicio TEXT NOT NULL DEFAULT '20:00',
  horario_nocturno_fin TEXT NOT NULL DEFAULT '07:00',
  nivel_vocabulario TEXT NOT NULL DEFAULT 'basico',
  tamano_texto TEXT NOT NULL DEFAULT 'normal',
  alto_contraste INTEGER NOT NULL DEFAULT 0,
  fuente_dislexia INTEGER NOT NULL DEFAULT 0,
  creado_en TEXT NOT NULL
);

CREATE TABLE conversaciones (
  id TEXT PRIMARY KEY,
  profile_id TEXT NOT NULL REFERENCES profiles(id),
  iniciada_en TEXT NOT NULL,
  finalizada_en TEXT
);

CREATE TABLE mensajes (
  id TEXT PRIMARY KEY,
  conversacion_id TEXT NOT NULL REFERENCES conversaciones(id),
  origen TEXT NOT NULL CHECK (origen IN ('nina','asistente')),
  texto TEXT NOT NULL,
  via TEXT NOT NULL CHECK (via IN ('texto','voz')),
  timestamp TEXT NOT NULL
);

CREATE TABLE eventos_seguridad (
  id TEXT PRIMARY KEY,
  mensaje_id TEXT NOT NULL REFERENCES mensajes(id),
  tipo TEXT NOT NULL CHECK (tipo IN ('crisis_detector','juez_seguridad')),
  categoria TEXT,                          -- alinear con las 10 categorías DPO, sección 2
  accion TEXT NOT NULL CHECK (accion IN ('bloqueado','redirigido')),
  timestamp TEXT NOT NULL
);

CREATE TABLE panel_auth (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  pin_hash TEXT NOT NULL                   -- nunca texto plano, usar bcrypt o equivalente
);
```

---

## 7. Sistema de diseño — el "efecto Gemini"

### 7.1 Paleta de color (identidad propia, no copiada)

| Uso | Color | Nota |
|---|---|---|
| Fondo principal | `#F4F7FB` | azul-gris muy claro |
| Texto principal | `#2D3648` | evitar negro puro |
| Orbe — base | gradiente `#6C9BF5` → `#B18CF0` | identidad propia, no replica la de ningún producto |
| Estado `listening` | acento `#3ED0C4` mezclado en el gradiente | |
| Estado `thinking` | mismo gradiente, rotación más rápida | |
| Estado `speaking` | gradiente pulsando con la amplitud del audio | |
| Estado `winding_down` | `#F5C97A` → `#F2A65A` | cálido, transmite calma |
| Estado `blocked` | `#B8C4D9` (gris-azulado neutro) | nunca rojo |
| Estado `error` | `#E8A33D` (ámbar) | no rojo saturado |

### 7.2 Técnica del orbe orgánico (Canvas 2D + ruido simplex)

No existe una librería "lista para usar" que recree este efecto — se construye con estas
piezas, todas ligeras y aptas para móvil (se descarta WebGL/Three.js como base por
rendimiento en celular y por ser más frágil para que un agente autónomo lo depure sin
supervisión constante; ver 7.8 para una ruta de mejora opcional más adelante):

1. Definir **8 puntos de control** distribuidos uniformemente en un círculo.
2. En cada frame, desplazar el radio de cada punto usando `simplex-noise` muestreado en
   `(cos(ángulo), sin(ángulo), tiempo)` — esto da una deformación orgánica y continua, sin
   saltos ni repeticiones obvias.
3. Conectar los puntos con una curva suave (Catmull-Rom convertida a Bézier) para evitar
   aristas — un polígono de 8 puntos sin suavizado se ve geométrico, no orgánico.
4. Rellenar con un gradiente radial o cónico usando los colores de 7.1, generado con
   `chroma-js` para que las transiciones de color entre estados sean interpolaciones
   perceptualmente suaves, no un corte brusco.
5. Dibujar una segunda copia desenfocada detrás (`ctx.filter = 'blur(Npx)'`) para el
   efecto de resplandor/glow.

### 7.3 Reactividad de audio en tiempo real

- **Estado `listening`:** conectar el `MediaStream` de `getUserMedia` a un `AnalyserNode`
  (`fftSize: 256`). En cada frame, leer `getByteFrequencyData()`, promediar, normalizar a
  0-1, y usar ese valor para modular la escala y la amplitud de ruido del paso 7.2.
- **Estado `speaking`:** mismo mecanismo, pero la fuente es la reproducción del audio TTS
  (conectar el elemento de audio o `AudioBufferSourceNode` a su propio `AnalyserNode`), no
  el micrófono.
- **Estados sin audio real** (`idle`, `thinking`, `winding_down`, `blocked`, `error`): el
  valor de "volumen" se simula con una onda seno lenta, para que el orbe nunca esté
  completamente estático.

### 7.4 Parámetros visuales por estado (punto de partida — ajustable tras ver el resultado real)

| Estado | Escala base | Amplitud de ruido | Velocidad de rotación | Reactivo a audio | Blur/glow |
|---|---|---|---|---|---|
| `idle` | 1.0, "respiración" ±3% cada 4s | 0.15 | 6°/s | No | 24px, opacidad 0.4 |
| `listening` | 1.0–1.25 según volumen | 0.15–0.6 según volumen | 12°/s | Sí (mic) | 32px, opacidad 0.6 |
| `thinking` | 1.05 fijo | 0.35 constante | 40°/s | No | 28px, pulso de opacidad 0.4↔0.7 cada 800ms |
| `speaking` | 1.0–1.3 según amplitud TTS | 0.2–0.7 según amplitud | 15°/s | Sí (TTS) | 36px, opacidad 0.7 |
| `winding_down` | 0.9 fijo | 0.1 | 4°/s | No | 20px, opacidad 0.3, transición de color 2s |
| `blocked` | 0.85 fijo | 0.05 | 2°/s | No | 16px, opacidad 0.25 |
| `error` | 0.9, temblor ±2% cada 200ms | 0.1 | 3°/s | No | 18px, opacidad 0.3 |

### 7.5 Tipografía y movimiento

- Familia: Nunito, Baloo 2 o Quicksand (Google Fonts). `<!-- TODO: elegir una en el Hito 1 -->`
- Tamaño base 18px mínimo, máximo 3 niveles de jerarquía.
- Transiciones de UI (Framer Motion, no el orbe): 300-500ms, `ease-in-out`. Respetar
  `prefers-reduced-motion`.

### 7.6 Vidrio esmerilado (glassmorphism) en la UI secundaria

- Barra de texto, botones de control y tarjetas de perfil usan `backdrop-filter: blur(20px)`
  sobre un fondo semitransparente (`rgba(255,255,255,0.6)`) — refuerza la sensación de que
  el orbe "vive" detrás de una superficie de vidrio, consistente con el lenguaje visual que
  se busca imitar sin copiar assets.

### 7.7 Iconografía

- Ilustraciones simples y redondeadas, sin realismo, sin fotos de personas reales.
- Ningún ícono ni imagen debe usar personajes con derechos de autor.

### 7.8 Ruta de mejora opcional (no bloqueante)

Si después del Hito 1 quieres acercarte aún más a un look 3D/volumétrico, la evolución
natural es `three.js` + `@react-three/fiber` con un shader de ruido en el fragment shader
en vez de Canvas 2D. **No es parte del alcance base** porque introduce riesgo real de
depuración para un agente autónomo (errores de shader son difíciles de verificar solo con
tests) y mayor consumo de batería en móvil. Considerarlo únicamente como una iteración
posterior, con la versión Canvas 2D ya funcionando y aprobada por ti.

---

## 8. Máquina de estados

Implementada con **XState v5**, no con booleans sueltos ni un `switch` informal — la
definición formal de estados y transiciones reduce directamente el riesgo de que el agente
deje un estado inalcanzable o una transición inconsistente.

| Estado | Disparador |
|---|---|
| `idle` | Sin actividad |
| `listening` | VAD detecta voz o la niña mantiene presionado el botón de micrófono |
| `thinking` | Audio/texto enviado, esperando respuesta del LLM |
| `speaking` | TTS está reproduciendo audio |
| `winding_down` | Faltan 20-30 min para el límite de uso diario (sección 9.5) |
| `blocked` | El detector de crisis o el juez de seguridad intervino |
| `error` | Falla de conexión con backend/LLM |

`winding_down` nunca interrumpe una respuesta en curso — solo se evalúa la transición entre
turnos. Cada estado, al activarse, aplica los parámetros visuales de la tabla 7.4 al `VoiceOrb`.

---

## 9. Seguridad y contenido — requisito obligatorio, no opcional

- **Todo** turno de conversación (texto o voz transcrita) debe pasar por el detector de crisis
  pre-LLM **antes** de llegar al modelo. El pipeline de voz no puede tener un camino directo
  STT → LLM que lo evite por razones de latencia.
- **Toda** respuesta del LLM debe pasar por el juez de seguridad **antes** de convertirse a voz
  o mostrarse en texto.
- Los mensajes de bloqueo (`blocked`) deben ser amigables y predefinidos, no generados
  dinámicamente por el LLM en ese momento.
- Esta sección se revisa manualmente por ti antes de dar por cerrado cualquier hito que la toque.

### 9.1 Identidad de la IA siempre visible
La interfaz debe recordar de forma recurrente (no solo en el primer uso) que se trata de una
IA, no de una persona.

### 9.2 Toggle de conversación IA, apagado por defecto
Función que activas explícitamente desde el panel parental, separada de cualquier función no
conversacional (módulos educativos, cuentos).

### 9.3 Lista de temas y palabras bloqueadas (configurable)
Editable desde el panel parental (sección 11), capa **adicional** antes del juez de seguridad.
`<!-- TODO: definir lista inicial de temas/palabras -->`

### 9.4 Sin roleplay de relación personal
La IA nunca se presenta como amiga, nunca fomenta secretos entre ella y la niña. Aplica al
system prompt y a cualquier persona/tono configurado.

### 9.5 Límite de tiempo diario y modo nocturno
Configurable por perfil (sección 10), con `winding_down` avisando 20-30 min antes del corte.
`<!-- TODO: confirmar límite diario y horario nocturno por defecto (sugerencia de partida: 30-45 min entre semana, horario nocturno 20:00-07:00) -->`

### 9.6 Relación con las categorías ya alineadas por DPO

El modelo (sección 2) ya fue entrenado para negar amablemente y redirigir en 10 categorías de
riesgo. El detector de crisis y el juez de seguridad **no reemplazan eso** — existen porque
(a) un modelo de 7B puede fallar en casos límite pese al DPO, y (b) dan un registro auditable
para el panel parental que el comportamiento "aprendido" del modelo no puede dar por sí solo.
Al registrar un evento de seguridad (tabla `eventos_seguridad`, sección 6.3), usa las mismas
10 categorías del dataset DPO como taxonomía (`categoria`), para que el panel parental hable
el mismo lenguaje que el entrenamiento.

---

## 10. Perfiles y personalización

- Cada niña tiene su propio perfil: nombre/apodo, edad, avatar ilustrado, color de acento
  personal (de la paleta 7.1).
- Calibración de vocabulario/complejidad por edad, aplicada al contexto enviado al LLM.
- Intereses opcionales, usados solo para personalizar sugerencias del módulo educativo
  (sección 12) — nunca compartidos fuera de la app.
- Límite de tiempo diario y horario nocturno individual (9.5).
- Selección de perfil simple y visual, sin contraseña — el PIN es exclusivo del panel
  parental (11.1).

---

## 11. Panel de administración parental

### 11.1 Autenticación
PIN o contraseña propia del padre (hash con bcrypt o equivalente, tabla `panel_auth`), ruta
no accesible por accidente desde la vista de las niñas.

### 11.2 Vista de conversaciones
Historial completo por perfil, transcripción de texto de cada turno (de la tabla `mensajes`),
filtrable por niña y fecha. El audio original no se guarda por defecto.

### 11.3 Eventos de seguridad
Vista separada de la tabla `eventos_seguridad`, con el turno completo de contexto y la
categoría DPO correspondiente (9.6).

### 11.4 Ajustes por perfil
Editar límite de tiempo, horario nocturno, lista de bloqueo (9.3), toggle de IA (9.2), sin
tocar archivos de configuración a mano.

### 11.5 Retención de datos
Periodo configurable (sugerido: 90 días) tras el cual se eliminan automáticamente los
registros de `conversaciones`/`mensajes`. Exportable antes de expirar.
`<!-- TODO: confirmar o ajustar el periodo de retención -->`

---

## 12. Módulos educativos y de aprendizaje

### 12.1 Modo guiado (estilo socrático)
Ante preguntas de tarea, guía con preguntas hacia la respuesta en vez de darla directo, salvo
que la niña la pida explícitamente más de una vez.

### 12.2 Honestidad epistémica calibrada para niños
Si el modelo no está seguro de un dato, lo dice con lenguaje simple ("no estoy segura, vamos
a buscarlo juntas") en vez de inventar.

### 12.3 Modo cuentos y lectura en voz alta
Usa el TTS ya integrado para narrar cuentos cortos.

### 12.4 Actividades sugeridas
Basadas en los intereses del perfil (sección 10).

### 12.5 Principio anti-diseño-adictivo (obligatorio)
Cualquier gamificación premia curiosidad o completar una actividad, nunca tiempo en la app.
Prohibido: notificaciones de retorno, countdowns de urgencia, scroll infinito.

---

## 13. Accesibilidad

- Tamaño de texto ajustable por perfil (normal/grande/muy grande).
- Alto contraste opcional.
- Fuente alternativa para dislexia (Lexend u OpenDyslexic), seleccionable por perfil.
- Subtítulos de texto sincronizados mientras el TTS habla (usa el mismo stream de
  `transcript_partial` del contrato WebSocket, sección 6.2).

---

## 14. Requisitos no funcionales

- **Red:** acceso solo dentro de la red local/tailnet. HTTPS obligatorio para micrófono en
  navegadores móviles (`tailscale serve`, no exponer al WAN).
- **Privacidad:** cero llamadas salientes a servicios externos. Persistencia 100% local.
- **Rendimiento objetivo:** dado que el modelo (4.9GB) cabe completo en los 16GB de VRAM sin
  offload a CPU, el objetivo recomendado es **<1.5s** desde que la niña termina de hablar
  hasta que empieza a escucharse la respuesta — más agresivo que si usaras un modelo grande
  con offload. Ajustable tras medir en el Hito 3.
- **Dispositivos objetivo:** navegador móvil (iOS/Android) en red local, orientación vertical.

---

## 15. Hitos y criterios de aceptación

Cada hito se entrega como commit(s) separado(s). No se avanza al siguiente hito sin que el
anterior cumpla su "Definición de Terminado".

### Hito 1 — Scaffold visual + orbe estilo Gemini

**Setup:**
- `npm create vite@latest sirius -- --template react-ts`
- Instalar: `framer-motion`, `xstate`, `@xstate/react`, `simplex-noise`, `chroma-js`
  (+ `@types/chroma-js`).
- Estructura de carpetas exactamente como sección 6.1.

**Sistema de diseño:**
- Variables CSS de 7.1 en `src/styles/tokens.css`, importadas globalmente — cero colores
  hardcodeados fuera de este archivo.
- Fuente elegida (7.5) cargada vía Google Fonts o `@fontsource`.

**El orbe:**
- `VoiceOrb.tsx`: Canvas 2D standalone, técnica de 7.2, con un oscilador seno simulando
  volumen (todavía sin audio real ni backend).
- Props: `state: OrbState`, `audioLevel: number`.
- Aplica los parámetros numéricos de la tabla 7.4 según el estado.

**Máquina de estados:**
- FSM de la sección 8 implementada con XState.
- Los 7 estados disparables con botones de debug temporales.

**Definición de terminado:**
- `npm run build` sin errores, `npm run dev` navegable.
- El orbe se renderiza y anima incluso sin backend.
- Los 7 estados producen resultados visualmente distinguibles entre sí (no solo el color:
  la velocidad/amplitud también deben notarse).
- Revisión tuya confirmando que el orbe se ve orgánico, no como un polígono rígido.

### Hito 2 — Chat de texto real
- Conexión al LLM leída desde `.env` (4.3) — nunca hardcodeada.
- Contrato WebSocket de la sección 6.2 implementado en ambos lados.
- Integración del detector de crisis y el juez de seguridad en el flujo de texto.
- Cada turno se escribe en las tablas `conversaciones`/`mensajes` (6.3).
- **Definición de terminado:** conversación de texto funcional de extremo a extremo, un caso
  de prueba que dispare `blocked` correctamente y quede registrado en `eventos_seguridad`,
  y una prueba manual de que cambiar el `.env` cambia el modelo usado sin tocar código.

### Hito 3 — Backend de voz (Pipecat)
- Pipeline: transporte → Silero VAD → faster-whisper → [detector de crisis] → LLM →
  [juez de seguridad] → TTS → transporte. Los dos checkpoints de seguridad como
  `FrameProcessor` explícitos en la definición del pipeline, no funciones ad-hoc.
- Expuesto vía WebSocket con el contrato de 6.2.
- Medir la latencia real extremo-a-extremo y compararla contra el objetivo de la sección 14.
- **Definición de terminado:** conversación de voz completa funciona por script de prueba,
  sin frontend todavía. Latencia medida y documentada en el README.

### Hito 4 — Integración frontend-voz
- El frontend consume `audio_chunk` y `state_change` reales del backend (6.2).
- `useAudioAnalyser.ts` conectado al micrófono real (7.3), reemplazando el oscilador simulado
  del Hito 1.
- **Definición de terminado:** conversación de voz completa desde el navegador de escritorio,
  el orbe reacciona a volumen real tanto en `listening` como en `speaking`.

### Hito 5 — Acceso móvil
- HTTPS vía `tailscale serve`, probado desde un celular real.
- Ajustes de UI mobile-first (tamaños de toque ≥44px, orientación vertical).
- **Definición de terminado:** conversación de voz completa desde un celular, permiso de
  micrófono concedido correctamente.

### Hito 6 — Perfiles por niña
- `ProfileSelector.tsx` + tabla `profiles` (6.3).
- Calibración de vocabulario conectada al contexto enviado al LLM según perfil activo.
- **Definición de terminado:** crear/editar/seleccionar perfiles funciona; límite de tiempo
  y horario nocturno se aplican correctamente por perfil.

### Hito 7 — Panel de administración parental
- Autenticación (11.1) con PIN hasheado.
- `ConversationLog.tsx` (11.2), `SafetyEventsLog.tsx` (11.3) con filtros por niña/fecha.
- `ProfileSettings.tsx` (11.4) editando la tabla `profiles` sin tocar archivos.
- Job de retención de datos (11.5) aplicado automáticamente.
- **Definición de terminado:** cada elemento probado manualmente, incluyendo un evento de
  seguridad generado a propósito y verificado en el panel con su categoría DPO correcta.

### Hito 8 — Módulos educativos y de aprendizaje
- Modo socrático (12.1), honestidad epistémica (12.2), modo cuentos (12.3), actividades por
  interés (12.4) — vía ajustes al system prompt, documentados en el README.
- **Definición de terminado:** un caso de cada módulo probado, checklist anti-diseño-adictivo
  (12.5) revisado y aprobado.

### Hito 9 — Accesibilidad
- Escala de texto, alto contraste, fuente para dislexia, subtítulos sincronizados (sección 13).
- **Definición de terminado:** cada opción probada visualmente en al menos un dispositivo real.

### Hito 10 — Endurecimiento y pruebas finales
- Manejo de errores de red/reconexión (estado `error`).
- Casos límite de seguridad (mensajes ambiguos, silencios largos, interrupciones).
- Revisión cruzada de que ningún módulo nuevo abrió una grieta en la sección 9.
- Retirar o esconder tras una bandera de desarrollo los botones de debug del Hito 1.
- **Definición de terminado:** checklist manual completo de la sección 9 revisado y aprobado.

---

## 16. Reglas operativas para el agente autónomo

1. Un commit por sub-tarea, con mensaje descriptivo. No mezclar cambios de hitos distintos.
2. Si una dependencia de la sección 4 no está disponible o falla su instalación, detenerse y
   reportarlo — no sustituirla sin avisar.
3. Si una instrucción es ambigua, dejar `<!-- TODO(agente): -->` y usar la interpretación más
   conservadora, nunca inventar alcance nuevo.
4. Mantener un `README.md` actualizado con: cómo levantar el proyecto, variables de entorno,
   estado de cada hito, y la fuente elegida (7.5) y latencia medida (Hito 3).
5. No tocar la sección 9 (seguridad) sin marcarlo explícitamente para revisión humana.
6. No desviarse de los contratos de datos de la sección 6 sin actualizarlos aquí primero.
7. No inventar valores de la sección 7 — si falta un parámetro, usar el más cercano ya
   definido y dejar el TODO correspondiente.

---

## 17. Cómo usar este documento con OpenHands

- Colócalo en la raíz del repositorio antes de abrir la primera conversación.
- Pégale al agente solo el hito en el que estás trabajando (ej. "Implementa el Hito 1 según
  PROJECT_SPEC.md, sección 15"), no todo el documento como tarea única.
- Al cerrar cada hito, revisa el diff/commit antes de continuar con el siguiente.
- Para el Hito 3, 7, 8 y 10 en particular (voz, panel parental, módulos educativos y
  seguridad), revisa el código manualmente además de correr los tests.
- Mientras OpenHands construye, Ornith está cargado en llama-server para el propio agente —
  independiente del `.env` de la sección 4.3, que es lo que la app *terminada* usa para
  hablar con el modelo kidsafe. No necesitas ambos modelos cargados a la vez.

---

## 18. Referencias de mercado (por qué existen las reglas de la sección 9 y 12)

- **Miko** (robot educativo 5-10 años): conversación IA apagada por defecto, lista de temas
  bloqueados, sin retención de grabaciones de voz. Base de 9.2 y 9.3.
- **Khanmigo** (tutor de Khan Academy): guía con preguntas en vez de dar la respuesta directa,
  transparencia activa hacia quien supervisa. Base de 9.1, 11 y 12.1.
- **Kinzoo/Kai**: principio explícito de no simular ser humano ni hacer roleplay de relación
  personal. Base de 9.4.
- **Character.AI**: eliminó el chat abierto para menores de 18 años en noviembre de 2025, tras
  demandas relacionadas con el suicidio de un menor y el hallazgo de personajes que
  impersonaban figuras dañinas. La causa raíz señalada fue el patrón de chat abierto tipo
  "amigo/compañero" sin los límites de 9.1-9.4. Referencia de qué evitar, no de qué imitar.
- **Amazon Alexa Kids**: recordatorios de "bajar el ritmo" antes del corte por hora de dormir
  (base de `winding_down` y 9.5). Amazon fue sancionado por la FTC con 25 millones de dólares
  por retener grabaciones de voz de menores pese a solicitudes de eliminación — recordatorio
  de por qué "cero llamadas salientes" (14) y la retención configurable (11.5) son la
  diferencia estructural más importante entre este proyecto y todos los productos comerciales
  revisados.
- **Legislación reciente sobre IA y menores** (ej. SB 243 en California): exige identidad de
  IA visible, recordatorios periódicos, protocolos de crisis, y prohíbe el diseño "adictivo".
  Base de 9.1 y 12.5.
