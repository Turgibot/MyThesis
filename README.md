# MyThesis
My master thesis in the open university of Israel at the NBEL lab


## DataSet
The dataset is artificially produced using a system that combines the Mujoco physics simulator and a Unity's rendering engine.
The User selects the recording preferences in the unity GUI.
Another feature is a randomizer that produces a selected number of randomized scenes. 

### User options:

1. Default Object type: Cube, Sphere, Tetrahedron, Torus, Cofee Mug, Capusle, Spinner.
2. Default Object Color: select from a color picker.
3. Default Size: Scaling Factor of object's dimentions in the range [0.5, 3].
4. Default Background texture. Select from 5 options [1, 5].
5. Default Speed: Scaling Factor of the speed of the robot movement. select a value in the range [0.5, 3].
6. Positive threshold of event camera. Selected from the range [0, 2].
7. Negative threshold of event camera. Selected from the range [0, 2].
8. Lighting Intensity. Selected from the range [0.1, 1].
9. Lighting Angle. Selected from the range [0.1, 0.5] *PI.
10. Recording Duration in seconds.

Each recording produces a pickle file. The name of the file holds the selected recording preferences in this format:

<type>_<color>_<size>_<speed>_<pos_th>_<neg_th>_<light_intencity>_<light_angle>_<date>_<time>
For example,

cube_#FF0000_1_1_3_0.12_0.08_0.55_0.45_2_08022022_125836

### Randomizer
When using a randomizer, The scene properties are controlled by a script. The user can state the number of recordings and the duration of each individual recording.
All other preferences are selected randomly in an even distribution manner.
