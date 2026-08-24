using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using System.Linq;


public class Agent412 : Agent
{
    GUIStyle guiStyle = new GUIStyle();
    [SerializeField] private GameObject PrisonCube;
    [SerializeField] private GameObject coinPref;
    [SerializeField] private Transform Env;
    [SerializeField] private bool stochastic_actions = false;

    [System.Serializable]
    private class SerializableFloats
    {
        /** This class is created so we can manipulate the distribution
         * list from the Unity Editor as a List, since arrays are not Serializable. **/
        public float[] distributionOfLevel =  new float[3];
    }

    /**
     * distributions_list : is defined via Unity's editor.
     * locations_list : Used to initialize all possible locations of coins.
     * clone_list : List to detect active coins.
     * **/
    [SerializeField] private List<SerializableFloats> distributions_list = new List<SerializableFloats>();
    private List<List<Vector3>> locations_list = new List<List<Vector3>>();
    private List<GameObject> clone_list = new List<GameObject>();


    // step size of actions :
    public float step = 5f;
    public int levels  = 4;
    private int curr_level;

    private Vector3 initPos;
    private Quaternion initRot;
    private Vector3 tempPosition;

    private Vector3 mid_pos;


    public override void Initialize()
    {
         /**
          * In this function we initialize positions that determine
          * the possible spawn locations of coins (InitPossibleLocations()).
          * **/

        base.Initialize();

        initPos = gameObject.transform.localPosition + Vector3.right*step;
        initRot = gameObject.transform.localRotation;
        mid_pos = initPos;
        tempPosition = gameObject.transform.localPosition;
        InitPossibleLocations();

    }

    // This function is called at the beginning of every episode.
    public override void OnEpisodeBegin()
    {

        base.OnEpisodeBegin();
        // Destroy remaining coins from previous episode.
        DestroyCoins(clone_list);

        // Activate Negative Coin if collected in previous episode.
        if(!PrisonCube.activeSelf) PrisonCube.SetActive(true);
        curr_level = 0;

        // Reposition our agent.
        gameObject.transform.localRotation = initRot;
        gameObject.transform.localPosition = tempPosition;
        mid_pos = initPos;

        // Spawn Coins according to the distribution set in *distributions_list*
        RandomCoinSpawn(locations_list, distributions_list , clone_list);

    }

    // Called whenever the Decision Requester component requests an action.
    // It depends on the Decision Period.
    public override void OnActionReceived(ActionBuffers actions)
    {

        base.OnActionReceived(actions);

        /**
         * Use bool stochastic_actions. If true then an action
         * has an Pr[error] = 0.1 as described in the report.
         * If false then the action is received as it was sent.
         * **/

        if (curr_level > levels) EndEpisode();
        curr_level++;
        mid_pos = new Vector3(gameObject.transform.position.x + step, gameObject.transform.position.y, initPos.z);
        if (!stochastic_actions)
        {
            // actions correspond to tensors of dqn_mlagents.py file
            if (actions.DiscreteActions[0] == 1) // tensor[1] --> up
                gameObject.transform.position = mid_pos + Vector3.forward * 4f;
            else if (actions.DiscreteActions[0] == 0) // tensor[0] --> mid
                gameObject.transform.position = mid_pos;
            else if (actions.DiscreteActions[0] == 2) // tensor[2] --> bot
                gameObject.transform.position = mid_pos - Vector3.forward * 4f;
        }
        else
        {
            // With error of moving Pr{a_i} = 0.9 , Pr{a_i + up} = 0.1  , [up + up = down]
            int rand_i = Random.Range(1, 11);
            if (actions.DiscreteActions[0] == 1)
            {
                if (rand_i < 9)
                    gameObject.transform.position = mid_pos + Vector3.forward * 4f;
                else
                    gameObject.transform.position = mid_pos + Vector3.forward * (-4f);
            }
            else if (actions.DiscreteActions[0] == 0)
            {
                if (rand_i < 9)
                    gameObject.transform.position = mid_pos;
                else
                    gameObject.transform.position = mid_pos + Vector3.forward * 4f;
            }
            else if (actions.DiscreteActions[0] == 2)
            {
                if (rand_i < 9)
                   gameObject.transform.position = mid_pos - Vector3.forward * 4f;
                else
                    gameObject.transform.position = mid_pos;
            }
        }

    }


    // After every Step() it collects observations
    public override void CollectObservations(VectorSensor sensor)
    {
        base.CollectObservations(sensor);

        // For this env states are simple.
        // Agent's position on plane (don't need position.y):
        sensor.AddObservation(gameObject.transform.position.x);

    }


    // Using Unity's collision detection tools & add rewards accordingly.
    private void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.CompareTag("Coin"))
        {
            Destroy(collision.gameObject);
            AddReward(1f);
        }
        if (collision.gameObject.CompareTag("Prison"))
        {
            PrisonCube.SetActive(false);
            AddReward(-1f);
        }
    }


    // Used for testing whether actions perform as expected, using the keyboard.
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        base.Heuristic(actionsOut);

        int heuristic_action = -1;
        if (Input.GetKey(KeyCode.W)) heuristic_action = 0;
        else if (Input.GetKey(KeyCode.S)) heuristic_action = 1;
        else if (Input.GetKey(KeyCode.D)) heuristic_action = 2;

        ActionSegment<int> discrete_act = actionsOut.DiscreteActions;
        discrete_act[0] = heuristic_action;
    }


    /**
     * Spawns coins according to our distributions_list using pseudorandom
     * numbers. Probabilities can be up to one decimal digit. **/
    private void RandomCoinSpawn(List<List<Vector3>> _list , List<SerializableFloats> x_distribution, List<GameObject> clone_p)
    {
        List<Vector3> tempAccess;
        for (int i = 0; i < levels; i++)
        {

            tempAccess = _list[i];
            // arrange distribution values. :
            float a = x_distribution[i].distributionOfLevel[0] * 10;
            float b = x_distribution[i].distributionOfLevel[1] * 10 + a;
            // Random call
            int rand = Random.Range(1, 11);

            Vector3 clone_transform;
            // Choose position according to rand
            if (rand <= a)
                clone_transform = tempAccess[0];
            else if (rand > a && rand <= b)
                clone_transform = tempAccess[1];
            else
                clone_transform = tempAccess[2];

            // Spawn our clone.
            GameObject clone_temp = Instantiate(coinPref, clone_transform, coinPref.transform.rotation, parent: Env);
            clone_p.Add(clone_temp);
        }
    }

    // Destroy remaining coins:
    private void DestroyCoins(List<GameObject> coins)
    {
        for(int i = 0; i < coins.Count; i++)
        {
            if (coins[i] != null) Destroy(coins[i]);
        }
    }


    private void InitPossibleLocations()
    {
        // Initialize positions with a List of Vector3s. The list length equals
        // the number of levels; each Vector3 corresponds to positions (x,y,z) of that level
        Vector3 temp_pos = new Vector3(mid_pos.x + step, mid_pos.y , mid_pos.z);
        for(int i = 0; i < levels; i++)
        {

            locations_list.Add(new List<Vector3>());
            for (int j = -1; j < 2; j++)
            {
                locations_list[i].Add(temp_pos + Vector3.forward * (float)j * 4f);
            }

            temp_pos = new Vector3(temp_pos.x + step,temp_pos.y, temp_pos.z);
        }
    }


    // Display some information.
    private void OnGUI()
    {
        guiStyle.fontSize = 50;
        guiStyle.normal.textColor = Color.white;
        GUI.Label(new Rect(10, 10, 100, 20), "Current Level " + curr_level, guiStyle);
        GUI.Label(new Rect(10, 60, 100, 20), "Cumulative Reward " + GetCumulativeReward(), guiStyle);
    }

}
