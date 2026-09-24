# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Web app: Vite + React + TypeScript + three.js (@react-three/fiber, drei) in `studio/web`, served by a Python FastAPI backend (`studio/server`) that runs MuJoCo physics and the brain and streams body poses over a WebSocket. Chosen by the user from three offered options.

## Users

The researcher (a university student) building "One Brain, Many Bodies". Two situations, both confirmed:
- **Research:** day-to-day experiments on a MacBook Air M3: connecting and disconnecting bodies, choosing what to train, watching bodies babble, damaging actuators, checking live numbers.
- **Demos:** showing the system live to professors or an audience (laptop screen or projector): speaking or typing a goal like "find the apple" and watching several bodies act as one mind.

## Product Purpose

Bio-Bot Studio is the control room for a self-learning brain wired like the fruit fly's nervous system (MaleCNS). It shows one shared 3D "free world" in which realistic bodies live together, and lets the researcher control what the brain is connected to, what it trains, and what goal it pursues. Success: an experiment can be set up, run, observed and explained from one screen, and a live demo tells the two research stories without narration from a slide.

## Positioning

The two research questions the studio exists to show:
1. **Self-learning:** a body connected to the brain with unnamed channels discovers what it is and how to move; no manual configuration.
2. **Coordination:** several different bodies connected to one brain act as one mind toward a single goal.

## Operating Context

- Runs locally: `studio/server` (Python, MuJoCo) + browser. Heavy training happens on Kaggle's free GPUs; the studio will launch and monitor those jobs.
- Two maps. **Sandbox:** an empty stage for testing bodies one at a time, then two, then all. **Open world:** an 80 × 80 m map with a furnished house (a low cabinet only the fly fits under), a road loop with sidewalks and street lamps, a downtown of six buildings, stairs and a ramp, a park, a lake with water physics, and hills. The apple can be placed at named spots on either map.
- Bodies: Unitree G1 humanoid, Unitree Go2 dog, Skydio X2 drone, and flybody fruit fly enlarged 100x with physically consistent scaling (all from MuJoCo Menagerie), plus a go-kart built for this project.
- Controllers today: off (limp, disconnected), hold (stand / hover / brake), babble (random motor exploration), manual (the researcher sets each joint, or drives the car / flies the drone). The drone's hover controller and the car's pure-pursuit controller can reach named places. These are "teachers", not the brain.

## Capabilities and Constraints

- **Stage 0 brain is built and running** (untrained): the MaleCNS connectome (166,700 neurons, 25.58M connections) as a rate network with neuron fatigue, on the Mac GPU at 20 Hz. The bodies' joint movements feed its sense neurons; it does not move the bodies. The studio's Brain tab shows it live in 3D.
- Training stages still to build: 1 Teach the bodies one by one → 2 Together → 3 Remove the teachers → 4 Keep learning alone. The studio must show these truthfully as locked or pending until they exist, never as working.
- Command box is a rule-based parser today (placeholder for speech → local LLM → goal vector).
- Physics has 1.24x real-time headroom in the open world with all five bodies (2.35x in the sandbox); the fly needs a 1 ms physics step.
- Budget is $0: no paid services.
- Terminology in the UI (plain words): "body", "Stand" / "Hover", "Relax", "Explore", "Motor power", "Put back", "Take control" / "Drive it" / "Fly it", "Brain", "Listening". In code and docs: modes hold / off / babble, actuator strength, stages.

## Brand Commitments

- Name: **Bio-Bot Studio** (repo: github.com/Karang1908/bio-bot).
- Credit lines for third-party models must stay visible somewhere: Unitree G1 and Go2 (BSD-3-Clause, Unitree Robotics), Skydio X2 (Apache-2.0), flybody (Janelia / Google DeepMind, Apache-2.0), MaleCNS connectome (Janelia FlyEM, CC-BY 4.0).

## Evidence on Hand

- Real models and the running simulation; measured numbers in `Requirements.md`.
- No results from training exist yet. The brain activity shown is real (the untrained connectome responding to the bodies' senses). Do not show fabricated learning curves or accuracies.

## Product Principles

1. **Truthful instrument.** Show only what is real; pending features are visibly pending, with the reason.
2. **The world is the hero.** The 3D scene carries the story; controls support it without covering it.
3. **One screen, two audiences.** Plain words first (the user found the dense pixel-font version too hard to understand); details open on demand.
4. **Bodies are peers.** Every body gets the same controls and the same visual weight, and is identified by its name and its own colour (user-confirmed).
