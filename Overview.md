# One Brain, Many Bodies: The Idea in Plain Words

_Non-technical overview · 23 September 2026_

---

## The short version

We are building an artificial brain that you can plug into almost anything (a robot, a drone, a computer, a house full of smart devices) and it **works out for itself** what it has been connected to and how to use it. Nobody has to set it up.

Then we go one step further: we plug **several bodies into the same brain at once** and give it one goal. The bodies work together because they are not a team of separate robots talking to each other. They are one mind in several bodies.

So the project asks two questions:

1. **Self-learning.** Can a machine configure itself, so that nobody has to set up a robot, a piece of software or a smart home by hand?
2. **Coordination.** When one brain lives in several different bodies at once, can it use all of them together toward one goal?

---

## Part 1: The problem with machines today

Every machine you own had to be configured by a person.

- A **robot** is calibrated in the factory and trained for its exact body. Swap a leg for a slightly different one, or let a motor wear out, and it often needs recalibrating or retraining.
- A **smart home** only does what you told it. Someone has to connect each device, name it, and write rules like *"if the motion sensor triggers after 10 pm, turn on the hallway light."*
- **Computers and servers** run on scripts and settings that people write and maintain.

In every case a human sits in the middle, translating between what the machine *has* and what we *want*. That's slow, it's expensive, and it breaks whenever something changes.

Animals don't work like that. A newborn foal is standing within hours. Nobody told it which muscle is which. It tried things, felt what happened, and learned. We want machines that learn their own bodies the same way.

---

## Part 2: Research question 1 — a machine that configures itself

### What "self-learning" means here

The brain is given a set of **unnamed wires**. Some carry signals in (these are its senses) and some carry commands out (these are its muscles). It is **not told** things like "this is a camera", "this is the left knee", "this switch controls the kitchen light" or "this number is the room temperature."

It has to find out, the way a baby does:

1. **Try something.** Send a small command down one wire.
2. **Notice what changes.** Did any of the senses react?
3. **Remember the connection.** "When I do *this*, *that* happens."
4. **Build a picture of itself.** Over time it learns which wires belong together, which ones move it, and which ones tell it about the world.

We call that picture its **self-map**: the brain's own understanding of what it controls. Because the self-map is something we can measure, we can check it against the truth and watch it fill in as the brain explores.

### How it learns: a childhood, then a life

Learning happens in two stages, like growing up.

**Childhood (in simulation, with teachers).** The brain practises in a computer simulation on many different made-up bodies: some with two legs, some with four, some that fly, some that roll. Teachers help during this stage:
- a teacher that **explains what words mean** in terms of outcomes ("*run* means move along the ground fast and stay upright"; "*find the apple* means the apple is in view and within reach")
- a teacher that knows the hidden truth of the simulation and tells the brain how well it did
- sometimes a simple example controller the brain can copy to get started

The point of the childhood is not to learn one body. It is to **learn how to learn a body**: after practising on hundreds of bodies, a brand-new one is just another puzzle of the same kind.

**Adult life (on the real device, no teachers).** Before deployment we remove all the teachers. The brain keeps only what it learned, including its own sense of when it has done well. A fruit fly has this built in: a small set of neurons release dopamine, which means "that was good" or "that was bad," and that signal changes the fly's memory. Our brain gets the same mechanism. So even after the teachers are gone it keeps learning on the device, adapting to new bodies, wear and damage by itself, cheaply, without needing a big computer.

### Why this matters

If this works, you connect a new robot and it learns to use itself. You install a new smart plug and the system works out what it switched on. A motor weakens and the robot compensates without anyone noticing. **Setup becomes the machine's job, not yours.**

---

## Part 3: Research question 2 — one brain, many bodies

### The apple story

We teach the brain three bodies, one at a time:
- a **drone**, which is fast, flies, and sees a lot from above, but can't pick things up
- a **humanoid**, which is slow and walks, but can pick things up and carry them
- a **fly** (a real fruit-fly model enlarged 100 times), which is small and can get into gaps the others can't

Then we connect **all three to the same brain at once** and say: ***"Find the apple and bring it to the table."***

What we hope to see: the drone rises and scans the room. It spots the apple behind the sofa, and at that instant the humanoid is already turning and walking there, because the humanoid's "mind" is the same mind that just saw the apple. If the apple is hidden where only the small walker fits, the small walker goes in. The humanoid picks the apple up and carries it to the table.

No message was sent between the bodies. There was nothing to send.

### Why this is different (the "weirdly agentic" part)

Today, when several robots work together, each one has **its own brain**. They have to talk to each other ("I found it, it's at position X"), agree on who does what, and cope with delays and misunderstandings.

Here there is **one brain**. The drone's eyes are the humanoid's eyes. It is less like a team and more like a single creature with several bodies. Nature has a close example: an **octopus** keeps most of its neurons in its arms. Its central brain sets the goals, and each arm handles its own movement, but it is still one animal.

This raises questions nobody has really answered:
- Does the brain treat the three bodies as **one self or three**?
- When the drone's camera sees the humanoid walking, does the brain realise **"that's me"**? Think of a mirror test for a mind spread across bodies.
- Does it **focus** on one body at a time, like a person who drives on autopilot while concentrating on a conversation?
- If we **unplug** one body in the middle of a task, does it re-plan? If we **plug in** a new one, does it start using it?

### Levels of teamwork we'll look for

| Level | What you'd see |
|---|---|
| 1. Side by side | All three search on their own |
| 2. Shared seeing | One body sees the apple and the others instantly head there |
| 3. Division of labour | The drone scouts, the humanoid fetches, the small walker checks tight spaces |
| 4. Physical teamwork | Bodies help each other, e.g. the drone guides the humanoid around obstacles |
| 5. Self-recognition | The brain knows the figure in the drone's camera is its own body |

We expect levels 1 and 2 to appear almost as soon as the bodies are connected, because the brain shares one memory of the world. Levels 3 and 4 need a short period of practising together. Level 5 is an open question, and one of the most interesting parts of the project.

---

## Part 4: Why a fruit fly's brain?

Scientists have mapped every neuron and every connection in a male fruit fly's brain and nerve cord: about **166,700 neurons** and roughly **25 million connections**. That map is called a **connectome**, and it's free for anyone to use.

We use it as the *wiring diagram* for our artificial brain, for three reasons:
- **Evolution already designed it** for exactly our problem: sensing, moving, finding things, learning from reward, and adapting.
- It has a **brain** and a separate **nerve cord** (the fly's spinal cord). That maps neatly onto our one-brain-many-bodies design: one shared brain, plus a nerve cord for each body.
- It lets us ask a real scientific question: **does nature's wiring learn better than the generic designs used in AI today?**

We don't assume the fly's wiring is better. We test it against ordinary AI designs, including a version with the *same number of connections but shuffled randomly*. If the fly's real wiring wins, that's a discovery. If it doesn't, that's an honest result too.

---

## Part 5: How language fits in

You can speak to it: *"find the apple."*

1. Your voice is turned into text.
2. A language model turns the sentence into a simple goal: the **target** is an apple, the **action** is to find it and reach it.
3. That goal is handed to the brain, and **the brain decides how** each body achieves it. The drone flies, the humanoid walks, the small walker crawls.

During training, a teacher explains what each word *means* in terms of results. After training the brain keeps its own small vocabulary, so for the words it has learned it no longer needs the language model at all. Understanding a brand-new word still needs the teacher's help once.

---

## Part 6: Applications

### From self-learning (plug it in, it works out the rest)
- **Robots:** new robots, arms and drones that learn their own bodies without calibration, and keep adapting as parts wear out or get replaced.
- **Smart homes:** connect your devices and the system works out what each one does and learns your routines, with no rule-writing.
- **Computers and servers:** a small fleet of machines that learns which machine is best for which job and keeps things running, using only approved actions.
- **Buildings and factories:** heating, cooling, lighting and energy systems that fold in new sensors and equipment automatically.
- **Old or undocumented equipment:** it works out what unlabelled controls do by testing them safely.

### From coordination (one mind, many bodies)
- **Search and rescue:** a drone scouts from above, a ground robot reaches the person, and a small robot squeezes into rubble.
- **Help at home:** house cameras act as eyes and a home robot acts as hands. *"Where did I leave my keys?"*
- **Warehouses:** flying scanners and ground movers working as one system.
- **Farms and inspection:** drones spot a problem and ground machines fix it.
- **Mixed fleets:** add or remove machines mid-job without reprogramming anything.

### Safety: configure the fence, not the function
A machine that learns by trying things must not try dangerous things. So one thing is always set by a human: **the fence**. That means what the system may never do (touch the stove, unlock the front door, delete files), how fast it may move, and which actions are allowed at all. Inside the fence it learns everything by itself. It also starts by watching before acting, and prefers small, reversible actions first.

---

## Part 7: What the final demo looks like

1. **One brain, one body at a time.** Connect a body. At first it moves randomly, then it figures itself out and moves with purpose. A live picture of its self-map fills in on screen as it learns.
2. **All bodies at once.** Connect all three. Say *"find the apple."* Watch the bodies share what they see and split up the work.
3. **Damage.** Weaken one of the drone's motors mid-flight, or unplug the drone completely. Watch the brain adapt and re-plan.
4. **A new body.** Plug in a body it has never seen and watch it get absorbed into the team.
5. **Beyond robots.** Connect a few smart-home devices and watch it work out what each one does.

---

## Part 8: What's new here

- Other systems that control many kinds of robot are **told** what each joint is. Ours isn't.
- Other projects that put a fly connectome into a body only drive a **fly**. We test whether it helps with bodies it never evolved for.
- Other robot teams use **separate brains** that talk. We use **one brain** that shares everything.
- Smart homes, robots and software today need **manual setup**. Ours sets itself up.

We searched the research carefully but not exhaustively, so these claims will be double-checked before any publication.

---

## Part 9: What could go wrong (honestly)

- **Learning to walk from scratch is hard.** Humanoid walking is the hardest skill here, so it comes last, and it may need the copy-a-simple-controller head start.
- **The fly's wiring may not help** on bodies it didn't evolve for. That is still a valid research result.
- **Size mismatch.** A real fruit fly is 2.5 mm long, which is useless next to a human-sized robot in the same room. So in the shared world the fly model is enlarged 100 times, with its weight, strength and timing all scaled together so its physics stays consistent: a 25 cm fly that weighs about 1 kg.
- **Real-world learning is slow.** A real house gives the brain a few examples an hour, not millions. That's why childhood happens in simulation first.
- **Teaching itself the wrong thing.** Once the teachers are gone, the brain judges its own success. On very unusual bodies that judgement may drift, so we measure it and can bring a teacher back briefly.

---

## Part 10: The plan in stages

1. **Build the brain** from the fly's wiring map, and make sure it runs stably.
2. **The 3D worlds** (built: Bio-Bot Studio). A sandbox for testing one body at a time, and a small open world with a house, roads and sidewalks, a downtown block, stairs, a park, a lake and hills. There are five realistic bodies, including a go-kart you can drive. You can inspect each body and control it by hand, and give commands in a command box. Both research questions are tested here first.
3. **Swap in the fly wiring** and compare it with ordinary AI designs.
4. **Add self-learning:** the dopamine-style learning, the self-map, damage tests, and removing the teachers.
5. **Harder bodies and more of them:** the drone, dog and fly first, many generated practice bodies, and finally the humanoid.
6. **Language:** voice commands and the brain's own vocabulary.
7. **Beyond robots:** smart-home devices and computers, inside a safety fence.

---

## Part 11: Cost

**Free.** Development runs on a MacBook Air, and heavy training runs on Kaggle's free GPUs. The fly connectome, the physics simulator and all the software are free and open. See [Requirements.md](Requirements.md).

---

## Glossary

| Word | Meaning |
|---|---|
| **Connectome** | A complete wiring map of a nervous system: every neuron and who connects to whom |
| **MaleCNS** | The connectome of a male fruit fly's brain and nerve cord, published by Janelia Research Campus |
| **Body** | Anything the brain is connected to: a robot, a drone, a house, a computer |
| **Channel** | One unnamed wire into or out of the brain: a single sense or a single "muscle" |
| **Self-map** | The brain's learned picture of which of its commands affect which of its senses |
| **Nerve cord** | The fly's version of a spinal cord. In our design each body gets its own copy, and all of them share one brain |
| **Dopamine signal** | The brain's built-in "that was good / bad" message, which drives learning after the teachers are gone |
| **Teacher** | A helper used only during training (for example, one that explains what "run" means), removed before deployment |
| **Childhood / adult life** | Practising in simulation with teachers, then learning alone on a real device |
| **Fence** | The few rules a human sets once to keep self-learning safe |
