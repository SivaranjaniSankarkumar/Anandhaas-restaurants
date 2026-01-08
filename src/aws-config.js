import { Amplify } from 'aws-amplify';

const awsConfig = {
  Auth: {
    Cognito: {
      userPoolId: 'us-east-1_ZZfX9CLwE',
      userPoolClientId: '30n0mub75mec2n4ei7v7jj4tjl',
      region: 'us-east-1'
    }
  }
};

Amplify.configure(awsConfig);

export default awsConfig;