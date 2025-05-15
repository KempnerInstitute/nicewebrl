# nicewebrl
Python library for quickly making interactive RL Apps with [NiceGUI](https://nicegui.io/). It is particularly suited for hooking up [JAX](https://github.com/google/jax) based RL environments to web interfaces. JAX is useful for blazing fast iteration on AI algorithms. With this library, you can use the exact same environment for human subject experiments.

<img src="media/comparing.jpeg" alt="Comparison Image" style="width: 100%; max-width: 800px;">

# Install

```bash
# pip install
pip install git+https://github.com/wcarvalho/nicewebrl

# more manually (first clone then)
conda create -n nicewebrl python=3.10 pip wheel -y
conda activate nicewebrl
pip install -e .
```


# Getting Started

### Example 1: Craftax
We've provided an example of making a web app to have humans control a craftax agent in `examples/craftax`.

<img src="https://raw.githubusercontent.com/MichaelTMatthews/Craftax/main/images/building.gif" alt="Craftax" style="width: 100%; max-width: 200px;">

**(1) Install nicewebrl with examples**
```bash
pip install "git+https://github.com/wcarvalho/nicewebrl.git#egg=nicewebrl[craftax]"
# or 
pip install -e ".[craftax]"
```

**(2) run the web app from the directory**
```bash
cd examples/crafter
python web_app.py
```
**NOTE**: this will make a `data` and `.nicegui` folder which you can delete to remove user data.

**COMING SOON**: Instructions for putting this on a remote server.

### Example 2: Overcooked
![Overcooked Environment](https://github.com/FLAIROx/JaxMARL/blob/main/docs/imgs/cramped_room.gif?raw=true)

**Make sure you have CMake installed**
```bash
pip install "git+https://github.com/wcarvalho/nicewebrl.git#egg=nicewebrl[jaxmarl]"
# or 
pip install -e ".[jaxmarl]"

cd examples/overcooked
python web_app.py
```

# Jax environments

The following are all Jax environments which can be used with this framework to run human subject experiments. The power of using jax is that one can use the **exact** same environment for human subjects experiments as for developing modern machine learning algorithms (especially reinforcement learning algorithms).

When targetting normative solutions, one may want to study algorithms asymptotic behavior with a lot of data. Jax makes it cheap to do this. nicewebrl makes it easy to compare these algorithms to human subject behavior.
<table style="width:100%; border-collapse: collapse;">
  <tr style="max-height: 150px; overflow: hidden;">
    <td style="border: 1px solid black; padding: 10px; text-align: center;">
      <a href="https://github.com/MichaelTMatthews/Craftax" target="_blank" style="text-decoration: none; color: inherit;">
        <center><strong>Craftax</strong></center>
      </a><br>
      <a href="https://github.com/MichaelTMatthews/Craftax" target="_blank">
        <img src="https://raw.githubusercontent.com/MichaelTMatthews/Craftax/main/images/building.gif" alt="Craftax" style="width: 100%; max-width: 400px;">
      </a>
      <p>This is a grid-world version of minecraft. </p>
    </td>
    <td style="border: 1px solid black; padding: 10px; text-align: center;">
      <a href="https://github.com/wcarvalho/JaxHouseMaze" target="_blank" style="text-decoration: none; color: inherit;">
        <center><strong>Housemaze</strong></center>
      </a><br>
      <a href="https://github.com/wcarvalho/JaxHouseMaze" target="_blank">
        <img src="https://github.com/wcarvalho/JaxHouseMaze/raw/main/example.png" alt="Housemaze" style="width: 100%; max-width: 400px;">
      </a>
      <p>This is a maze environment where new mazes can be easily be described with a string.</p>
    </td>
    <td style="border: 1px solid black; padding: 10px; text-align: center;">
      <a href="https://github.com/corl-team/xland-minigrid" target="_blank" style="text-decoration: none; color: inherit;">
        <center><strong>XLand-Minigrid</strong></center>
      </a><br>
      <a href="https://github.com/corl-team/xland-minigrid" target="_blank">
        <img src="https://github.com/corl-team/xland-minigrid/blob/main/figures/ruleset-example.jpg?raw=true" alt="XLand-Minigrid" style="width: 100%; max-width: 400px;">
      </a>
      <p>This environment allows for complex, nested compositional tasks. XLand-Minigrid comes with 3 benchmarks which together defnine 3 million unique tasks.</p>
    </td>
  </tr>
  <tr style="max-height: 150px; overflow: hidden;">
    <td style="border: 1px solid black; padding: 10px; text-align: center;">
      <a href="https://github.com/epignatelli/navix" target="_blank" style="text-decoration: none; color: inherit;">
        <center><strong>Navix</strong></center>
      </a><br>
      <a href="https://github.com/epignatelli/navix" target="_blank">
        <img src="https://minigrid.farama.org/_images/GoToObjectEnv.gif" alt="Navix" style="width: 100%; max-width: 400px;">
      </a>
      <p>This is a jax implementation of the popular Minigrid environment.</p>
    </td>
    <td style="border: 1px solid black; padding: 10px; text-align: center;">
      <a href="https://github.com/FLAIROx/JaxMARL" target="_blank" style="text-decoration: none; color: inherit;">
        <center><strong>Overcooked (multi-agent)</strong></center>
      </a><br>
      <a href="https://github.com/FLAIROx/JaxMARL" target="_blank">
        <img src="https://github.com/FLAIROx/JaxMARL/blob/main/docs/imgs/cramped_room.gif?raw=true" alt="Overcooked" style="width: 100%; max-width: 400px;">
      </a>
      <p>This is a popular multi-agent environment.</p>
    </td>
    <td style="border: 1px solid black; padding: 10px; text-align: center;">
      <a href="https://github.com/FLAIROx/JaxMARL" target="_blank" style="text-decoration: none; color: inherit;">
        <center><strong>STORM (multi-agent)</strong></center>
      </a><br>
      <a href="https://github.com/FLAIROx/JaxMARL" target="_blank">
        <img src="https://github.com/FLAIROx/JaxMARL/blob/main/docs/imgs/storm.gif?raw=true" alt="STORM" style="width: 100%; max-width: 400px;">
      </a>
      <p>This environment specifies Matrix games represented as grid world scenarios.</p>
    </td>
  </tr>
</table>




<table>
  <thead>
    <tr>
      <th>Name</th>
      <th>Domain</th>
      <th>Visualization</th>
      <th>Goal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>Craftax</code></td>
      <td>2D Minecraft</td>
      <td><img src="assets/craftax.gif" width="120"/></td>
      <td>Mine and craft resources in a Minecraft-like 2D world</td>
    </tr>
    <tr>
      <td><code>Kinetix</code></td>
      <td>2D Physics Control</td>
      <td><img src="assets/kinetix.gif" width="120"/></td>
      <td>Control rigid 2D bodies to perform dynamic tasks</td>
    </tr>
    <tr>
      <td><code>Navix</code></td>
      <td>MiniGrid</td>
      <td><img src="assets/navix.gif" width="120"/></td>
      <td>Navigate grid environments with JAX-based MiniGrid variant</td>
    </tr>
    <tr>
      <td><code>XLand–MiniGrid</code></td>
      <td>XLand</td>
      <td><img src="assets/xland_minigrid.gif" width="120"/></td>
      <td>Meta-RL tasks defined by dynamic rulesets</td>
    </tr>
    <tr>
      <td><code>JaxMARL</code></td>
      <td>Multi-agent RL</td>
      <td><img src="assets/jaxmarl.gif" width="120"/></td>
      <td>Cooperative and competitive multi-agent environments in JAX</td>
    </tr>
    <tr>
      <td><code>JaxGCRL</code></td>
      <td>Goal-Conditioned Robotics</td>
      <td><img src="assets/jaxgcrl.gif" width="120"/></td>
      <td>Goal-conditioned control in simulated robotics tasks</td>
    </tr>
    <tr>
      <td><code>Gymnax</code></td>
      <td>Classic RL</td>
      <td><img src="assets/gymnax.gif" width="120"/></td>
      <td>Classic control, bsuite, and MinAtar environments in JAX</td>
    </tr>
    <tr>
      <td><code>Jumanji</code></td>
      <td>Combinatorial</td>
      <td><img src="assets/jumanji.gif" width="120"/></td>
      <td>From simple games to NP-hard combinatorial problems</td>
    </tr>
    <tr>
      <td><code>Pgx</code></td>
      <td>Board Games</td>
      <td><img src="assets/pgx.gif" width="120"/></td>
      <td>Chess, Go, Shogi, and other turn-based strategy games</td>
    </tr>
    <tr>
      <td><code>Brax</code></td>
      <td>Physics Simulation</td>
      <td><img src="assets/brax.gif" width="120"/></td>
      <td>Differentiable physics engine for continuous control</td>
    </tr>
  </tbody>
</table>