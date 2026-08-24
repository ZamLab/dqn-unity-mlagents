using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine.UIElements;


public class Agent412p3 : Agent
{
    [SerializeField] private float force_boost;
    private RaycastHit fsensor0,fsensor1,fsensor2;
    private float wall_detect_distance0, coin_detect_distance0, wall_detect_distance1, coin_detect_distance1, wall_detect_distance2, coin_detect_distance2;
    private Vector3 sens0dir,sens1dir,sens2dir;
    private Vector3 init_pos;
    private Quaternion init_rot;
    [SerializeField] private Transform infEnvironment;
    [SerializeField] private GameObject coin_pref;
    [SerializeField] private List<Transform> coin_pos = new List<Transform>();
    private GameObject clone;
    private LayerMask _mask;
    private Rigidbody agent_body;
    private float distanceTravelled = 0;
    GUIStyle guiStyle = new GUIStyle();
    private bool yield_flag = false;
    private Vector3 temp_pos;


    /**
     * In the Update method we keep tracking values for every sensor.
     * A sensor is custom-built using Unity's SphereCasting.
     * It detects walls or coins according to the Tag that is set.
     *
     * For example, if the front sensor (fsensor0) detects a Coin: the
     * correlated coin-detection value (coin_detect_distance0) is assigned
     * the detection distance and the wall detector is assigned -1.
     * (Default is 0 so we set -1 to make it easier for the neural network.)
     *
     * Sensors:
     *  sensor0 -> Same direction as +z axis (Front).
     *  sensor1 -> Direction +45/2 degrees to the right side (+x).
     *  sensor2 -> Direction +45/2 degrees to the left side (+x).
     * **/
    void Update()
    {
        gameObject.transform.position += gameObject.transform.forward * force_boost * Time.deltaTime;
        sens0dir = gameObject.transform.forward;
        sens1dir = gameObject.transform.forward  + (gameObject.transform.forward + gameObject.transform.right);
        sens2dir = gameObject.transform.forward + (gameObject.transform.forward - gameObject.transform.right);

        if(!yield_flag) StartCoroutine(UpdateDistancePerTimer());


        if (Physics.SphereCast(gameObject.transform.localPosition, 0.35f, sens0dir, out fsensor0, 30f, _mask))
        {
            if (fsensor0.collider.gameObject.CompareTag("Wall"))
            {
                wall_detect_distance0 =fsensor0.distance;
                coin_detect_distance0 = -1;
            }
            if (fsensor0.collider.gameObject.CompareTag("Coin"))
            {
                wall_detect_distance0 = -1;
                coin_detect_distance0 = fsensor0.distance;
            }
        }

        if (Physics.SphereCast(gameObject.transform.localPosition, 0.35f, sens1dir, out fsensor1, 30f, _mask))
        {
            if (fsensor1.collider.gameObject.CompareTag("Wall"))
            {
                wall_detect_distance1 = fsensor1.distance;
                coin_detect_distance1 = -1;
            }
            if (fsensor1.collider.gameObject.CompareTag("Coin"))
            {
                wall_detect_distance1 = -1;
                coin_detect_distance1 = fsensor1.distance;
            }
        }

        if (Physics.SphereCast(gameObject.transform.localPosition, 0.35f, sens2dir, out fsensor2, 30f, _mask))
        {
            if (fsensor2.collider.gameObject.CompareTag("Wall"))
            {
                wall_detect_distance2 = fsensor1.distance;
                coin_detect_distance2 = -1;
            }
            if (fsensor2.collider.gameObject.CompareTag("Coin"))
            {
                wall_detect_distance2 = -1;
                coin_detect_distance2 = fsensor1.distance;
            }
        }

        // Debug sensors, only in Unity's Editor.
        Debug.DrawLine(gameObject.transform.localPosition, gameObject.transform.localPosition + sens0dir * fsensor0.distance,Color.blue);
        Debug.DrawLine(gameObject.transform.localPosition, gameObject.transform.localPosition + sens1dir * fsensor1.distance, Color.blue);
        Debug.DrawLine(gameObject.transform.localPosition, gameObject.transform.localPosition + sens2dir * fsensor2.distance, Color.blue);
    }

    private void OnCollisionEnter(Collision collision)
    {
        if (collision.collider.gameObject.CompareTag("Wall"))
        {
            AddReward(-1f); // smaller than coin
            EndEpisode();
        }
        else if (collision.collider.gameObject.CompareTag("Coin"))
        {
            AddReward(2f);
            EndEpisode();
        }
    }

    public override void OnEpisodeBegin()
    {
        base.OnEpisodeBegin();

        if (clone != null) Destroy(clone);
        // CLONE INSTANTIATE
        gameObject.transform.localPosition = init_pos;
        gameObject.transform.localRotation = init_rot;

        // Remove any force impact from previous episode:
        agent_body.rotation = init_rot;
        agent_body.velocity = Vector3.zero;

        // Spawn Coin :
        int rand_in = Random.Range(0, 4);
        clone = Instantiate(coin_pref, coin_pos[rand_in].transform.position, coin_pos[rand_in].transform.rotation, parent: infEnvironment);
        yield_flag = false;
        distanceTravelled = 0;

        temp_pos = init_pos;
    }


    public override void Initialize()
    {
        base.Initialize();

        init_pos = gameObject.transform.localPosition;
        init_rot = gameObject.transform.localRotation;
        // Layer masks used for sensors to define which objects in our scene can be sensed.
        _mask = LayerMask.GetMask("WallLayer", "CoinLayer");

        agent_body = gameObject.GetComponent<Rigidbody>();
    }


    public override void OnActionReceived(ActionBuffers actions)
    {
        base.OnActionReceived(actions);

        if (actions.DiscreteActions[0] == 0)
            gameObject.transform.Rotate(Vector3.zero);
        else if (actions.DiscreteActions[0] == 1)
            gameObject.transform.Rotate(Vector3.up*45f);
        else if (actions.DiscreteActions[0] == 2)
            gameObject.transform.Rotate(Vector3.up*(-45f));

        AddReward(distanceTravelled/1000);
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        base.CollectObservations(sensor);

        sensor.AddObservation(wall_detect_distance0);
        sensor.AddObservation(wall_detect_distance1);
        sensor.AddObservation(wall_detect_distance2);

        sensor.AddObservation(coin_detect_distance0);
        sensor.AddObservation(coin_detect_distance1);
        sensor.AddObservation(coin_detect_distance2);

        sensor.AddObservation(gameObject.transform.localPosition.x);
        sensor.AddObservation(gameObject.transform.localPosition.z);
    }

    private void OnGUI()
    {
        guiStyle.fontSize = 50;
        guiStyle.normal.textColor = Color.white;
        GUI.Label(new Rect(10, 10, 100, 20), "Distance Travelled " + distanceTravelled, guiStyle);
        GUI.Label(new Rect(10, 60, 100, 20), "Cumulative Reward " + GetCumulativeReward(), guiStyle);
    }

    /**
     * This coroutine is used to determine and increase the distance
     * travelled using small time steps.
     * **/
    private IEnumerator  UpdateDistancePerTimer()
    {
        yield_flag = true;
        yield return new WaitForSeconds(0.5f);
        distanceTravelled = Vector3.Distance(temp_pos, gameObject.transform.localPosition);
        temp_pos = gameObject.transform.localPosition;
        yield_flag = false;
    }
}
