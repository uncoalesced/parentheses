# Code of conduct

This page has two parts. The first is about the model: parentheses is uncensored on purpose, and anyone contributing needs to know that, because it decides which contributions fit. The second is about people: how we treat each other in issues, pull requests and email.

## Part 1: the model is uncensored

### What that means

parentheses is trained to answer what it's asked. Concretely:

- There is no refusal training. No data in any training stage teaches the model to decline a question because of its topic.
- There are no guard or safety models anywhere in the pipeline. They aren't used as teachers, as judges, or as filters on training data.
- There is no content filter in the chat script or the weights. What the model generates is what you see.
- It doesn't lecture. An answer shouldn't come wrapped in warnings, disclaimers or moral advice nobody asked for.

Uncensored doesn't mean careless with the truth. The other half of the goal is honesty: when the model doesn't know something, it should say "I don't know" instead of making something up. A model that refuses nothing but invents facts has failed just as badly as one that refuses everything. Getting both right at the same time is the hardest part of the project, and [the roadmap](docs/ROADMAP.md#teaching-it-to-say-i-dont-know) covers how I'm working on it.

### Why it matters to me

Refusals in small models mostly misfire. A 148-million-parameter model can't reliably tell a dangerous question from an ordinary one, so refusal training ends up blocking a medical question, a history question or a chemistry homework question because a word in it looked alarming. In testing, parentheses-0.9 refused none of 19 sensitive but legitimate technical questions. That's the behaviour I want to keep.

The person running the model should decide what's appropriate for their use, not the weights. Once a refusal is trained in, nobody downstream can take it out without retraining. A filter added on top can always be added later by whoever deploys the model; a refusal baked into the weights can't be removed by them. So the weights stay neutral and the choice stays with the user.

It also matters for the languages this project exists for. Safety datasets are overwhelmingly English. Training refusals from them into a Kannada or Tulu model teaches it English-centred judgements about what's acceptable to say, applied to languages and cultures those datasets never looked at.

And I'd rather be honest about what a model is. A small model that pretends to have judgement it doesn't have is worse than one that answers plainly and is clear about what it doesn't know.

### What this means for contributions

Contributions that add refusal behaviour, moralising, disclaimers, topic blocklists, or hidden filters to the model, its training data or the chat script won't be merged. That includes training data where the "preferred" answer is a refusal to a legitimate question.

Two rules still apply to what you contribute, because they're conditions of hosting the project, not limits on the model:

- What you submit has to be legal to host on GitHub and has to follow GitHub's terms.
- Training data has to pass the licence rule in [LICENSE_POLICY.md](LICENSE_POLICY.md): commercial-use licences only.

You're responsible for how you use the model. It will answer what you ask, and it will sometimes be confidently wrong. Check anything that matters.

## Part 2: how we treat each other

This part is short because it should be obvious, but writing it down means it's clear when it isn't followed.

### What I expect

- Disagree with ideas, not people. Technical arguments are welcome; personal attacks aren't.
- Assume good faith, especially across languages. Many people here write in English as a second or third language.
- Be patient with beginners. Someone asking how to install PyTorch is as welcome as someone sending a kernel patch.
- Credit other people's work, and be honest about where your data and code come from.

### What isn't acceptable

- Harassment, insults, slurs, or threats, in public or in private messages.
- Abuse aimed at anyone's language, caste, religion, region, gender, sexuality or nationality.
- Publishing someone's private information without their permission.
- Repeatedly derailing issues or pull requests after being asked to stop.

### Reporting

If someone's behaviour is a problem, email **mail@uncoalesced.com** with what happened and links if you have them. I'll read every report, reply to say I've received it, and keep your name private unless you tell me otherwise. Depending on what happened, I may ask the person to change how they're behaving, remove comments, or block them from the project. Right now I'm the only maintainer, so if the report is about me, say so in the email and I'll treat it with the same seriousness; you can also raise it publicly in an issue.

This applies in the project's GitHub repository, its issues and pull requests, and any email about the project.
