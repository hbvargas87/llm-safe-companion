# PROJECT_SPEC.md
## App de chat por texto y voz (estilo Gemini) — Modelo local SFT+DPO para uso infantil

> Este documento es el contrato de construcción para el agente autónomo (OpenHands + Ornith).
> Colócalo en la raíz del repo. Aliméntalo al agente **hito por hito**, no de una sola vez.

---

## 1. Resumen del proyecto

Construir una web app local (React + backend de voz) donde las usuarias (niñas, 6-12 años)
puedan hablar o escribir con un LLM local afinado (SFT+DPO) desde su celular, dentro de la
red doméstica, con una interfaz inspirada en Gemini: colores suaves, un elemento visual central
(orbe/onda) que reacciona en tiempo real a si la app está escuchando, pensando o hablando.

**Usuarias objetivo:** niñas de 6-12 años. La UI debe ser simple, sin texto denso, con
retroalimentación visual clara en cada estado.

---

## 2. Alcance y no-alcance

**Dentro de alcance:**
- Chat por texto y por voz con el modelo local vía llama-server.
- Interfaz visual con estados animados (idle / escuchando / pensando / hablando).
- Acceso desde celular dentro de la red local (vía Tailscale).
- Integración obligatoria con el pipeline de seguridad existente (detector de crisis + juez de seguridad).

**Fuera de alcance (el agente NO debe hacer esto sin preguntar):**
- Ninguna llamada a APIs externas (OpenAI, Google, ElevenLabs, etc.). Todo el procesamiento es local.
- Ninguna exposición a internet público (nada de puertos abiertos al WAN, nada de ngrok/cloudflared).
- No introducir autenticación de terceros, analytics, telemetría, ni CDNs externos más allá de los
  estrictamente necesarios para dependencias de build.
- No cambiar el modelo, el motor de inferencia (llama-server) ni el detector de seguridad existente.
- No decidir arquitectura de seguridad por su cuenta: si algo no está cubierto en la sección 6, debe
  detenerse y dejar un TODO explícito en vez de improvisar.
- No implementar la IA como un "amigo" o personaje que finge ser humano, ni roleplay de relación
  personal (ver sección 6.4). Este es un principio de diseño, no un detalle de copy.

---

## 3. Stack tecnológico (fijo, no negociable)

| Capa | Tecnología | Notas |
|---|---|---|
| Frontend | React + Vite + TypeScript | SPA, mobile-first |
| Animaciones | Framer Motion | para transiciones de estado y el orbe/onda central |
| Comunicación en vivo | WebSocket | frontend ↔ backend de voz, eventos de estado + audio |
| Orquestación de voz | Pipecat | pipeline STT → LLM → TTS |
| STT | faster-whisper | modelo tamaño a definir según latencia real (`<!-- TODO: elegir tamaño (base/small) tras medir latencia -->`) |
| VAD | Silero VAD | detección de inicio/fin de habla |
| TTS | Kokoro (preferido) o Piper (fallback si latencia/recursos lo exigen) | debe soportar español natural |
| LLM | Ornith vía llama-server, endpoint OpenAI-compatible | `<!-- TODO: pegar aquí URL:puerto actual, ej. http://<ip-pc-secundaria>:8081/v1 -->` |
| Seguridad | Detector de crisis pre-LLM + juez de seguridad (Gemma4) ya existentes | ver sección 6, es obligatorio |

No se permite sustituir ninguna pieza de esta tabla por "una alternativa más fácil de instalar"
sin dejarlo documentado como decisión explícita en el README, citando el motivo.

---

## 4. Arquitectura general

```
[Celular de la niña] --HTTPS/WSS (Tailscale)--> [Frontend Vite/React]
                                                        |
                                                        | WebSocket (audio + eventos de estado)
                                                        v
                                          [Backend Pipecat]
                                          Silero VAD -> faster-whisper (STT)
                                                |
                                                v
                                    [Detector de crisis pre-LLM] --(bloqueo si aplica)--> respuesta segura
                                                |
                                                v
                                    [Ornith vía llama-server] (LLM)
                                                |
                                                v
                                    [Juez de seguridad] --(bloqueo si aplica)--> respuesta segura
                                                |
                                                v
                                          Kokoro/Piper (TTS)
                                                |
                                                v
                                [Audio + evento "hablando"] -> Frontend
```

El chat de **texto** usa el mismo camino desde el detector de crisis en adelante, saltándose
solo STT/TTS.

---

## 5. Estados de UI y comportamiento visual

| Estado | Disparador | Comportamiento visual esperado |
|---|---|---|
| `idle` | Sin actividad | Orbe en reposo, animación sutil de "respiración" |
| `listening` | VAD detecta voz o usuario mantiene botón de micrófono | Orbe/onda reacciona al volumen en tiempo real |
| `thinking` | Audio/texto enviado, esperando respuesta del LLM | Animación de proceso (ej. pulso más rápido, sin reactividad a audio) |
| `speaking` | TTS está reproduciendo audio | Onda sincronizada con la amplitud del audio de salida |
| `winding_down` | Faltan 20-30 min para el límite de uso diario (ver 6.6) | Transición gradual de color/ritmo del orbe, tono calmado; aviso amigable de que la sesión terminará pronto |
| `blocked` | El detector de crisis o el juez de seguridad intervino | Estado visual neutro/calmado, sin alarmar; mensaje amigable predefinido |
| `error` | Falla de conexión con backend/LLM | Mensaje simple y amigable, sin jerga técnica, con opción de reintentar |

El agente debe implementar estos como una máquina de estados explícita (no booleans sueltos),
para que agregar estados futuros no rompa la lógica existente. `winding_down` no interrumpe una
conversación en curso — se activa entre turnos, nunca a mitad de una respuesta.

---

## 6. Seguridad y contenido — requisito obligatorio, no opcional

- **Todo** turno de conversación (texto o voz transcrita) debe pasar por el detector de crisis
  pre-LLM **antes** de llegar a Ornith. El pipeline de voz no puede tener un camino directo
  STT → LLM que lo evite por razones de latencia.
- **Toda** respuesta del LLM debe pasar por el juez de seguridad **antes** de convertirse a voz
  o mostrarse en texto.
- Si el agente autónomo necesita optimizar latencia del pipeline de voz, debe hacerlo en las
  etapas de audio (buffer sizes, streaming parcial), nunca quitando o esquivando estos dos
  checkpoints.
- Los mensajes de bloqueo (`blocked`) deben ser amigables y predefinidos, no generados
  dinámicamente por el LLM en ese momento.
- Esta sección se revisa manualmente por ti (no solo con tests automáticos) antes de dar por
  cerrado cualquier hito que la toque.

### 6.1 Identidad de la IA siempre visible
- La interfaz debe recordar de forma recurrente (no solo en el primer uso) que se trata de una IA,
  no de una persona. No es un aviso legal escondido; es un elemento visible del diseño.

### 6.2 Toggle de conversación IA, apagado por defecto
- La conversación abierta con el LLM es una función que tú activas explícitamente, separada de
  cualquier otra función no conversacional que la app pueda tener (juegos, cuentos, etc.).
- Debe poder apagarse con un solo toque, sin perder el resto de la app.

### 6.3 Lista de temas y palabras bloqueadas (configurable)
- Un archivo de configuración (no hardcodeado) con temas y palabras que el padre puede bloquear
  manualmente, como capa **adicional** antes del juez de seguridad — nunca como sustituto de él.
- `<!-- TODO: definir lista inicial de temas/palabras -->`

### 6.4 Sin roleplay de relación personal
- La IA nunca debe presentarse como amiga, actuar con afecto exclusivo, ni fomentar secretos entre
  ella y la niña ("esto quédatelo entre nosotras"). No hay excepción para hacerla "más cercana".
- Esto aplica tanto al system prompt de Ornith como a cualquier persona/tono que se le configure.

### 6.5 Tablero parental con historial y eventos de seguridad
- Vista simple, solo para ti, con: historial de conversaciones y cualquier evento donde se activó
  el detector de crisis o el juez de seguridad. No requiere ser elaborado — una tabla o log
  legible es suficiente en la primera versión.

### 6.6 Límite de tiempo diario y modo nocturno
- Límite de uso diario configurable, con el estado `winding_down` de la sección 5 avisando
  20-30 min antes del corte.
- Modo nocturno/horario de silencio configurable (la app no responde fuera de ese horario).
- `<!-- TODO: definir límite diario y horario nocturno -->`

### 6.7 Perfil por niña
- Si hay más de una niña usando la app, cada una debe tener su propio perfil con su propio
  límite de tiempo y, si aplica, calibración de vocabulario/complejidad por edad.

---

## 7. Requisitos no funcionales

- **Red:** acceso solo dentro de la red local / tailnet. HTTPS obligatorio para permitir
  micrófono en navegadores móviles (usar `tailscale serve` o certificado local, no exponer
  puertos al WAN).
- **Privacidad:** cero llamadas salientes a servicios externos en tiempo de ejecución.
- **Rendimiento objetivo:** `<!-- TODO: definir latencia máxima aceptable extremo-a-extremo, ej. <3s desde que termina de hablar hasta que empieza a escuchar la respuesta -->`.
- **Dispositivos objetivo:** navegador móvil (iOS/Android) en red local, orientación vertical.
- **Persistencia:** `<!-- TODO: decidir si se guarda historial de conversación y dónde (local, sin nube) -->`.

---

## 8. Hitos y criterios de aceptación

Cada hito se entrega como commit(s) separado(s). No se avanza al siguiente hito sin que el
anterior cumpla su "Definición de Terminado".

### Hito 1 — Scaffold visual (sin backend real)
- Proyecto Vite + React + TS inicializado, estructura de carpetas clara.
- Los 7 estados de la sección 5 implementados como máquina de estados, con datos simulados
  (mock) para poder ver las transiciones sin backend.
- **Definición de terminado:** `npm run build` sin errores, `npm run dev` navegable, los 7
  estados se pueden disparar manualmente (ej. botones de debug temporales).

### Hito 2 — Chat de texto real
- Conexión al endpoint de Ornith (llama-server) para texto plano.
- Integración del detector de crisis y el juez de seguridad en el flujo de texto.
- **Definición de terminado:** conversación de texto funcional de extremo a extremo, con al
  menos un caso de prueba que dispare el estado `blocked` correctamente.

### Hito 3 — Backend de voz (Pipecat)
- Pipeline Silero VAD → faster-whisper → (detector de crisis) → Ornith → (juez de seguridad) → TTS.
- Expuesto vía WebSocket con eventos de estado consumibles por el frontend.
- **Definición de terminado:** una conversación de voz completa funciona por línea de comandos
  o script de prueba, sin frontend todavía.

### Hito 4 — Integración frontend-voz
- El frontend consume audio y eventos de estado reales del backend de voz.
- Animaciones reactivas a volumen real (no simuladas).
- **Definición de terminado:** conversación de voz completa desde el navegador de escritorio.

### Hito 5 — Acceso móvil
- HTTPS vía Tailscale configurado, probado desde un celular real en la red.
- Ajustes de UI mobile-first (tamaños de toque, orientación).
- **Definición de terminado:** conversación de voz completa desde un celular, incluyendo permiso
  de micrófono concedido correctamente.

### Hito 6 — Endurecimiento y pruebas
- Manejo de errores de red/reconexión.
- Pruebas de los casos límite de seguridad (mensajes ambiguos, silencios largos, interrupciones).
- **Definición de terminado:** checklist manual de la sección 6 revisado y aprobado por ti.

### Hito 7 — Módulos parentales y de bienestar
- Toggle de conversación IA (6.2), lista de temas/palabras bloqueadas (6.3), tablero parental (6.5).
- Límite de tiempo diario + modo nocturno + estado `winding_down` conectado a un reloj real (6.6).
- Perfiles por niña si aplica (6.7).
- **Definición de terminado:** cada módulo probado manualmente por ti con al menos un caso real;
  el tablero parental muestra correctamente un evento de bloqueo generado a propósito.

---

## 9. Reglas operativas para el agente autónomo

1. Un commit por sub-tarea, con mensaje descriptivo. No mezclar cambios de hitos distintos.
2. Si una dependencia de la sección 3 no está disponible o falla su instalación, detenerse y
   reportarlo — no sustituirla por otra sin avisar.
3. Si una instrucción de este documento es ambigua, dejar un comentario `<!-- TODO(agente): -->`
   y continuar con la interpretación más conservadora, nunca inventar alcance nuevo.
4. Mantener un `README.md` actualizado con: cómo levantar el proyecto, variables de entorno
   necesarias, y estado de cada hito.
5. No tocar la sección 6 (seguridad) sin marcarlo explícitamente para revisión humana.

---

## 10. Cómo usar este documento con OpenHands

- Colócalo en la raíz del repositorio antes de abrir la primera conversación.
- Pégale al agente solo el hito en el que estás trabajando (ej. "Implementa el Hito 1 según
  PROJECT_SPEC.md, sección 8"), no todo el documento como tarea única.
- Al cerrar cada hito, revisa el diff/commit antes de pedirle que continúe con el siguiente.
- Para el Hito 3, 6 y 7 en particular (voz, seguridad y módulos parentales), revisa el código
  manualmente además de correr los tests — no te bases solo en "los tests pasaron".

---

## 11. Referencias de mercado (por qué existen las reglas de la sección 6)

No son reglas arbitrarias — cada una responde a algo observado en productos reales:

- **Miko** (robot educativo 5-10 años): conversación IA apagada por defecto, lista de temas
  bloqueados, sin retención de grabaciones de voz. Base de 6.2 y 6.3.
- **Khanmigo** (tutor de Khan Academy): transparencia activa y visible hacia quien supervisa,
  no un ajuste escondido. Base de 6.1 y 6.5.
- **Kinzoo/Kai**: principio explícito de no simular ser humano ni hacer roleplay de relación
  personal, porque confunde a los niños y puede fomentar apego poco saludable. Base de 6.4.
- **Character.AI**: tuvo que eliminar por completo el chat abierto para menores de 18 años en
  noviembre de 2025, tras demandas relacionadas con el suicidio de un menor y el hallazgo de
  personajes que impersonaban a figuras dañinas. La causa raíz señalada fue precisamente el
  patrón de chat abierto tipo "amigo/compañero" sin los límites de 6.1-6.4. Es la referencia
  de qué evitar, no de qué imitar.
- **Amazon Alexa Kids**: recordatorios de "bajar el ritmo" antes del corte por hora de dormir.
  Base del estado `winding_down` (5) y de 6.6. Nota aparte: Amazon fue sancionado por la FTC con
  25 millones de dólares por retener grabaciones de voz de menores pese a solicitudes de
  eliminación — un recordatorio de por qué el requisito de "cero llamadas salientes" de la
  sección 7 no es solo preferencia técnica tuya, sino la diferencia estructural más importante
  entre este proyecto y todos los productos comerciales revisados: ninguno de ellos procesa el
  audio 100% local.
