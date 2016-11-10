using UnityEngine;
using System.Collections;
using UnityEngine.UI;
using UnityEngine.SceneManagement;

public class TitleController : MonoBehaviour {

	public Text highScoreLabel;

	public void OnStartButtonClicked() {
		SceneManager.LoadScene ("Main");
	}

	// Use this for initialization
	void Start () {
		highScoreLabel.text = "Hi Score : " + PlayerPrefs.GetInt ("HightScore") + "m";
	}
	
	// Update is called once per frame
	void Update () {
	
	}
}
